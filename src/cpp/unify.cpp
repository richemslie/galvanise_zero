float PuctEvaluator::priorScore(PuctNode* node, int depth) const {

    float prior_score = node->getFinalScore(node->lead_role_index);
    if (node->visits > 8) {
        const PuctNodeChild* best = this->chooseTopVisits(node);
        if (best->to_node != nullptr) {
            prior_score = best->to_node->getCurrentScore(node->lead_role_index);
        }
    }

    float fpu_reduction = depth == 0 ? this->conf->fpu_prior_discount_root : this->conf->fpu_prior_discount;

    if (fpu_reduction > 0) {

        // original value from network / or terminal value
        float total_policy_visited = 0.0;

        for (int ii=0; ii<node->num_children; ii++) {
            PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
            if (c->to_node != nullptr && c->to_node->visits > 0) {
                total_policy_visited += c->policy_prob;
            }
        }

        fpu_reduction *= std::sqrt(total_policy_visited);
        prior_score -= fpu_reduction;
    }

    return prior_score;
}


void PuctEvaluator::setDirichletNoise(PuctNode* node) {

    // this->conf->dirichlet_noise_alpha < 0 - off
    // node->dirichlet_noise_set - means set already for this node
    // node->num_children < 2 - not worth setting
    if (node->dirichlet_noise_set ||
        node->num_children < 2 ||
        this->conf->dirichlet_noise_alpha < 0) {
        return;
    }

    // calculate noise_alpha based on number of children - credit KataGo & LZ0
    // note this is what I have manually been doing by hand for max number of children

    // magic number is 10.83f = 0.03 ∗ 361 - as per AG0 paper

    // XXX conf->dirichlet_noise_alpha now deprecated
    const float dirichlet_noise_alpha = 10.83f / node->num_children;

    std::gamma_distribution<float> gamma(dirichlet_noise_alpha, 1.0f);

    std::vector <float> dirichlet_noise;
    dirichlet_noise.resize(node->num_children, 0.0f);

    float total_noise = 0.0f;
    for (int ii=0; ii<node->num_children; ii++) {
        const float noise = gamma(this->rng);
        dirichlet_noise[ii] = noise;
        total_noise += noise;
    }

    // fail if we didn't produce any noise
    if (total_noise < std::numeric_limits<float>::min()) {
        return;
    }

    // normalize to a distribution
    for (int ii=0; ii<node->num_children; ii++) {
        dirichlet_noise[ii] /= total_noise;
    }

    // variable noise percent between 0.25 and this->conf->dirichlet_noise_pct, of > 0.25 to start with
    float noise_pct = this->conf->dirichlet_noise_pct;
    if (noise_pct > 0.25f) {
        noise_pct = 0.25f + this->rng.get() * (noise_pct - 0.25f);
    }

    // reset the policy with orig (with a temp)
    // new_dist = [x ** temp for x in d]

    if (node->num_children > 6) {

        // safety net
        for (int ii=0; ii<5; ii++) {

            float max_prob = -1;
            for (int ii=0; ii<node->num_children; ii++) {
                PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
                max_prob = std::max(max_prob, c->policy_prob);
            }

            if (max_prob < 0.15) {
                break;
            }

            float total_policy = 0;

            // apply temperature
            const float TEMP = 0.75;
            for (int ii=0; ii<node->num_children; ii++) {
                PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
                c->policy_prob = std::pow(c->policy_prob, TEMP);
                total_policy += c->policy_prob;
            }

            // normalise
            for (int ii=0; ii<node->num_children; ii++) {
                PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
                c->policy_prob /= total_policy;
            }
        }
    }

    // replace the policy_prob on the node
    float total_policy = 0;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
        c->policy_prob = (1.0f - noise_pct) * c->policy_prob + noise_pct * dirichlet_noise[ii];
        total_policy += c->policy_prob;
    }

    // re-normalize node (XXX shouldn't need to... look into later):
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
        c->policy_prob /= total_policy;
    }

    node->dirichlet_noise_set = true;
}


void PuctEvaluator::setPuctConstant(PuctNode* node, int depth) const {
    // XXX configurable
    const float cpuct_base_id = 19652.0f;
    const float puct_constant = depth == 0 ? this->conf->puct_constant_root : this->conf->puct_constant;

    //node->puct_constant = std::log((1 + node->visits + cpuct_base_id) / cpuct_base_id);
    float multiplier = std::max(1.0, (4.0 - depth) / 1.5);
    node->puct_constant = multiplier * std::log((1 + node->visits + cpuct_base_id) / cpuct_base_id);
    node->puct_constant += puct_constant;
}

float PuctEvaluator::getTemperature(int depth) const {
    if (depth >= this->conf->depth_temperature_stop) {
        return -1;
    }

    ASSERT(this->conf->temperature > 0);

    float multiplier = 1.0f + ((depth - this->conf->depth_temperature_start) *
                               this->conf->depth_temperature_increment);

    multiplier = std::max(1.0f, multiplier);

    return std::min(this->conf->temperature * multiplier, this->conf->depth_temperature_max);
}

const PuctNodeChild* PuctEvaluator::choose(const PuctNode* node) {
    const PuctNodeChild* choice = nullptr;
    switch (this->conf->choose) {
        case ChooseFn::choose_top_visits:
            choice = this->chooseTopVisits(node);
            break;
        case ChooseFn::choose_temperature:
            choice = this->chooseTemperature(node);
            break;
        default:
            K273::l_warning("this->conf->choose unsupported - falling back to choose_top_visits");
            choice = this->chooseTopVisits(node);
            break;
    }

    return choice;
}

bool PuctEvaluator::converged(int count) const {
    auto children = PuctNode::sortedChildren(this->root, this->sm->getRoleCount());

    if (children.size() >= 2) {
        PuctNode* n0 = children[0]->to_node;
        PuctNode* n1 = children[1]->to_node;

        if (n0 != nullptr && n1 != nullptr) {
            const int role_index = this->root->lead_role_index;

            if (n0->getCurrentScore(role_index) > n1->getCurrentScore(role_index)) {
                if (n0->visits > n1->visits + count) {
                    return true;
                }
            }
        }

        return false;
    }

    return true;
}
