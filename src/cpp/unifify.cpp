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
    if (this->conf->dirichlet_noise_alpha < 0) {
        return;
    }

    if (node->dirichlet_noise_set) {
        return;
    }

    const auto debug_noise = false;

    std::gamma_distribution<float> gamma(this->conf->dirichlet_noise_alpha, 1.0f);

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

    // normalize:
    for (int ii=0; ii<node->num_children; ii++) {
        dirichlet_noise[ii] /= total_noise;
    }

    const float noise_pct = this->conf->dirichlet_noise_pct;

    if (debug_noise) {
        PuctNode::dumpNode(node, nullptr, "before ... ", false, this->sm);
    }

    float total_policy = 0;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
        c->policy_prob = (1.0f - noise_pct) * c->policy_prob + noise_pct * dirichlet_noise[ii];
        total_policy += c->policy_prob;
    }

    // re-normalize node (XXX shouldn't need to... look into later):
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);
        c->policy_prob /= total_policy;
    }

    if (debug_noise) {
        PuctNode::dumpNode(node, nullptr, "after ... ", false, this->sm);
    }

    node->dirichlet_noise_set = true;
}

