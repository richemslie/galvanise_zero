
#include <unordered_set>

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

    // this->conf->dirichlet_noise_pct < 0 - off
    // node->dirichlet_noise_set - means set already for this node
    // node->num_children < 2 - not worth setting
    if (node->dirichlet_noise_set ||
        node->num_children < 2 ||
        this->conf->dirichlet_noise_pct < 0) {
        return;
    }

    // calculate noise_alpha based on number of children - credit KataGo & LZ0
    // note this is what I have manually been doing by hand for max number of children

    // magic number is 10.83f = 0.03 ∗ 361 - as per AG0 paper

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

    const float pct = this->conf->dirichlet_noise_pct;

    bool policy_squash = (this->conf->noise_policy_squash_pct > 0 &&
                          this->rng.get() > this->conf->noise_policy_squash_pct);
    if (policy_squash && node->getCurrentScore(node->lead_role_index) > 0.9) {
        policy_squash = false;
    }

    // replace the policy_prob on the node
    float total_policy = 0;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);

        // randomly reduce high fliers
        if (policy_squash) {
            c->policy_prob = std::min(this->conf->noise_policy_squash_prob,
                                      c->policy_prob);
        }

        c->policy_prob = (1.0f - pct) * c->policy_prob + pct * dirichlet_noise[ii];
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

    node->puct_constant = std::log((1 + node->visits + cpuct_base_id) / cpuct_base_id);
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

            if (n0->getCurrentScore(role_index) > n1->getCurrentScore(role_index) &&
                n0->visits > n1->visits + count) {
                return true;
            }
        }

        return false;
    }

    return true;
}

void PuctEvaluator::checkDrawStates(const PuctNode* node, PuctNode* next) {
    if (next->num_children == 0) {
        return;
    }

    // only support 2 roles
    ASSERT (this->sm->getRoleCount() == 2);

    auto legalSet = [](const PuctNode* node) {
        std::unordered_set <int> result;
        for (int ii=0; ii<node->num_children; ii++) {
            const PuctNodeChild* c = node->getNodeChild(2, ii);
            result.insert(c->move.get(node->lead_role_index));
        }

        return result;
    };

    // const int repetition_lookback_max = this->conf->repetition_lookback_max;
    const int repetition_lookback_max = 20;

    const int number_repeat_states_draw = this->conf->use_legals_count_draw;

    auto next_legal_set = legalSet(next);

    int repeat_count = 0;
    for (int ii=0; ii<repetition_lookback_max; ii++) {
        if (node == nullptr) {
            break;
        }

        if (node->lead_role_index == next->lead_role_index &&
            node->num_children == next->num_children &&
            next_legal_set == legalSet(node)) {
            repeat_count++;

            if (repeat_count == number_repeat_states_draw) {

                //K273::l_verbose("repeat state found at depth %d, legals %d",
                //                next->game_depth, (int) next_legal_set.size());


                for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
                    next->setCurrentScore(ii, 0.5);
                }

                next->is_finalised = true;
                next->force_terminal = true;

                return;
            }
        }

        node = node->parent;
    }
}

void PuctEvaluator::logDebug(const PuctNodeChild* choice_root) {
    PuctNode* cur = this->root;
    for (int ii=0; ii<this->conf->max_dump_depth; ii++) {
        std::string indent = "";
        for (int jj=ii-1; jj>=0; jj--) {
            if (jj > 0) {
                indent += "    ";
            } else {
                indent += ".   ";
            }
        }

        const PuctNodeChild* next_choice;

        if (cur->num_children == 0) {
            next_choice = nullptr;

        } else {
            if (cur == this->root) {
                next_choice = choice_root;
            } else {
                next_choice = this->chooseTopVisits(cur);
            }
        }

        bool sort_by_next_probability = (cur == this->root &&
                                         this->conf->choose == ChooseFn::choose_temperature);


        // for side effects of displaying probabilities
        Children dist;
        if (cur->num_children > 0 && cur->visits > 0) {
            const float temperature = std::max(1.0f, this->getTemperature(cur->game_depth));
            if (cur->visits < 3) {
                dist = this->getProbabilities(cur, temperature, true);

            } else {
                dist = this->getProbabilities(cur, temperature, false);
            }
        }

        PuctNode::dumpNode(cur, next_choice, indent, sort_by_next_probability, this->sm);

        if (next_choice == nullptr || next_choice->to_node == nullptr) {
            break;
        }

        cur = next_choice->to_node;
    }
}

const PuctNodeChild* PuctEvaluator::chooseTemperature(const PuctNode* node) {

    // XXX do we need this check?
    if (node == nullptr) {
        node = this->root;
    }

    float temperature = this->getTemperature(node->game_depth);
    if (temperature < 0) {
        return this->chooseTopVisits(node);
    }

    // subtle: when the visits is very low, we want to use the policy part of the
    // distribution - not the visits.
    Children dist;
    if (this->conf->dirichlet_noise_pct < 0 && node->visits < 3) {
        dist = this->getProbabilities(this->root, temperature, true);
    } else {
        dist = this->getProbabilities(this->root, temperature, false);
    }

    float expected_probability = this->rng.get() * this->conf->random_scale;

    if (this->conf->verbose) {
        K273::l_debug("temperature %.2f, expected_probability %.2f",
                      temperature, expected_probability);
    }

    float seen_probability = 0;
    for (const PuctNodeChild* c : dist) {
        seen_probability += c->next_prob;
        if (seen_probability > expected_probability) {
            return c;
        }
    }

    return dist.back();
}
