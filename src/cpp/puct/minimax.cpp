#include "minimax.h"

using namespace GGPZero;

PuctNodeChild* MiniMaxer::minimaxExpanded(PuctNode* node) {
    const int role_count = this->sm->getRoleCount();

    // returns the best child, or nullptr if there was no more minimax
    if (node->is_finalised) {
        return nullptr;
    }

    const int ri = node->lead_role_index;
    PuctNodeChild* best = nullptr;
    float best_score = -1;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(role_count, ii);

        if (child->use_minimax) {
            ASSERT(child->to_node != nullptr);

            PuctNodeChild* mc = minimaxExpanded(child->to_node);
            float score = -1;
            if (mc == nullptr) {
                score = child->to_node->getFinalScore(ri, true);
            } else {
                score = child->to_node->getCurrentScore(ri);
            }

            if (score > best_score) {
                best = child;
                best_score = score;
            }
        }
    }

    if (best != nullptr) {
        node->setCurrentScore(ri, best_score);
    }

    return best;
}

typedef std::vector <PuctNodeChild*> MiniChildren;

static MiniChildren sortTreeByPolicy(PuctNode* node, int role_count) {
    MiniChildren children;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(role_count, ii);
        children.push_back(child);
    }

    auto f = [](const PuctNodeChild* a, const PuctNodeChild* b) {
                 return a->policy_prob > b->policy_prob;
             };

    std::sort(children.begin(), children.end(), f);
    return children;
}

void MiniMaxer::expandTree(PuctNode* node, int depth) {
    const int role_count = this->sm->getRoleCount();

    if (depth == this->conf->minimax_specifier.size()) {
        return;
    }

    if (node->is_finalised) {
        return;
    }

    // get the specifier
    int policy_count = this->conf->minimax_specifier[depth].add_policy_count;

    // sort tree by policy
    auto children = ::sortTreeByPolicy(node, role_count);

    auto expand = [node, this](PuctNodeChild* child, int next_depth) {
       if (child->to_node == nullptr) {
           this->evaluator->expandChild(node, child);
       }

       ASSERT(child->to_node != nullptr);
       this->expandTree(child->to_node, next_depth);
    };

    int unexpanded_children = node->num_children;
    if (unexpanded_children <= this->conf->follow_max) {
        for (PuctNodeChild* c : children) {
            // note depth not decremented
            expand(c, depth);
            c->use_minimax = true;
        }

        return;
    }

    // add best from policy
    for (PuctNodeChild* c : children) {
        if (policy_count > 0) {
            expand(c, depth - 1);
            c->use_minimax = true;
            policy_count--;
            unexpanded_children--;
        } else {
            c->use_minimax = false;
        }
    }

    int random_count = std::min(unexpanded_children,
                                this->conf->minimax_specifier[depth].add_random_count);

    float chance_to_random_play_move = 1.0f / (unexpanded_children + 0.001f);

    while (random_count > 0) {
        for (PuctNodeChild* c : children) {
            if (c->use_minimax) {
                continue;
            }

            if (this->rng.get() < chance_to_random_play_move) {
                expand(c, depth - 1);
                c->use_minimax = true;
                random_count--;
            }
        }
    }
}
