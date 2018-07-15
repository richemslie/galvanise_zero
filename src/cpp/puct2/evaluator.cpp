/*

XXX
* shmem?  Could get a factor of 4 speed up (maybe).  Still will have the problem of needing even
  more non evaluated nodes.  At least can use multiple GPUs - which will help with turn around
  times of the node evaluations.
* (bugix - but inconsequential, I think) fix traversals > visits, when there is only one move

*/


#include "puct2/evaluator.h"
#include "puct2/node.h"

// for NodeRequestInterface
#include "scheduler.h"
#include "gdltransformer.h"

#include <k273/util.h>
#include <k273/logging.h>
#include <k273/strutils.h>
#include <k273/exception.h>

#include <cmath>
#include <climits>
#include <unistd.h>
#include <random>
#include <numeric>
#include <string>
#include <vector>
#include <tuple>

using namespace GGPZero::PuctV2;


///////////////////////////////////////////////////////////////////////////////

PathElement::PathElement(PuctNode* node, PuctNodeChild* choice,
                         PuctNodeChild* best, int num_children_expanded) :
    node(node),
    choice(choice),
    best(best),
    num_children_expanded(num_children_expanded) {
}

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::PuctEvaluator(GGPLib::StateMachineInterface* sm, NetworkScheduler* scheduler,
                             const GGPZero::GdlBasesTransformer* transformer) :
    conf(nullptr),
    sm(sm),
    basestate_expand_node(nullptr),
    scheduler(scheduler),
    game_depth(0),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0),
    do_playouts(false) {

    this->basestate_expand_node = this->sm->newBaseState();

    this->lookup = GGPLib::BaseState::makeMaskedMap <PuctNode*>(transformer->createHashMask(this->basestate_expand_node));
}

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::~PuctEvaluator() {
    free(this->basestate_expand_node);
    this->reset(0);
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::updateConf(const PuctConfig* conf) {
    if (conf->verbose) {
        K273::l_verbose("config verbose: %d, dump_depth: %d, choice: %s",
                        conf->verbose, conf->max_dump_depth,
                        (conf->choose == ChooseFn::choose_top_visits) ? "choose_top_visits" : "choose_temperature");

        K273::l_verbose("puct constant %.2f, %.2f/%.2f, root: %.2f/%.2f",
                        conf->puct_constant_init,
                        conf->puct_constant_min,
                        conf->puct_constant_max,
                        conf->puct_constant_min_root,
                        conf->puct_constant_max_root);

        K273::l_verbose("dirichlet_noise (alpha: %.2f, pct: %.2f), fpu_prior_discount: %.2f",
                        conf->dirichlet_noise_alpha, conf->dirichlet_noise_pct, conf->fpu_prior_discount);

        K273::l_verbose("temperature: %.2f, start(%d), stop(%d), incr(%.2f), max(%.2f) scale(%.2f)",
                        conf->temperature, conf->depth_temperature_start, conf->depth_temperature_stop,
                        conf->depth_temperature_max, conf->depth_temperature_increment, conf->random_scale);

        K273::l_verbose("scaled (at %d, reduce %.2f/%.2f). converge_ratio: %.2f, minimax (ratio %.2f, reqd_visits %d)",
                        conf->scaled_visits_at, conf->scaled_visits_reduce,
                        conf->scaled_visits_finalised_reduce,
                        conf->top_visits_best_guess_converge_ratio,
                        conf->minimax_backup_ratio, conf->minimax_required_visits);

        K273::l_verbose("think %.1f, relaxed %d/%d, policy_dilution_visits=%d, batch_size=%d",
                        conf->think_time, conf->converge_relaxed, conf->converge_non_relaxed,
                        conf->policy_dilution_visits, conf->batch_size);

        K273::l_verbose("expand_threshold_visits %d, #expansions_end_game %d",
                        conf->expand_threshold_visits, conf->number_of_expansions_end_game);
    }

    this->conf = conf;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::removeNode(PuctNode* node) {
    const GGPLib::BaseState* bs = node->getBaseState();
    this->lookup->erase(bs);
    this->node_allocated_memory -= node->allocated_size;

    free(node);
    this->number_of_nodes--;
}

void PuctEvaluator::releaseNodes(PuctNode* current) {
    // remove all children nodes if ref count == 0
    int role_count = this->sm->getRoleCount();
    for (int ii=0; ii<current->num_children; ii++) {
        PuctNodeChild* child = current->getNodeChild(role_count, ii);

        if (child->to_node != nullptr) {
            PuctNode* next_node = child->to_node;

            // wah a cycle...
            if (next_node->ref_count == 0) {
                K273::l_warning("A cycle was found in Player::releaseNodes() skipping");
                continue;
            }

            child->to_node = nullptr;

            ASSERT(next_node->ref_count > 0);
            next_node->ref_count--;
            if (next_node->ref_count == 0) {
                this->releaseNodes(next_node);
                this->garbage.push_back(next_node);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

PuctNode* PuctEvaluator::lookupNode(const GGPLib::BaseState* bs) {
    auto found = this->lookup->find(bs);
    if (found != this->lookup->end()) {
        PuctNode* result = found->second;
        result->ref_count++;
        return result;
    }

    return nullptr;
}

PuctNode* PuctEvaluator::createNode(PuctNode* parent, const GGPLib::BaseState* state) {

    // constraint sm, already set
    PuctNode* new_node = PuctNode::create(state, this->sm);

    // add to lookup table
    this->lookup->emplace(new_node->getBaseState(), new_node);

    // update stats
    this->number_of_nodes++;
    this->node_allocated_memory += new_node->allocated_size;

    new_node->parent = parent;
    if (parent != nullptr) {
        new_node->game_depth = parent->game_depth + 1;
        parent->num_children_expanded++;

    } else {
        new_node->game_depth = 0;
    }

    if (new_node->is_finalised) {
        // XXX ZZZ tmp hack to try and focus more on winning lines
        for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
            const float s = new_node->getCurrentScore(ii);
            if (s > 0.99) {
                new_node->setCurrentScore(ii, s * 1.05);
            }
        }


        return new_node;
    }

    // skips actually evaluation on nodes with only 1 child
    if (new_node->num_children == 1) {
        return new_node;
    }

    // goodbye kansas
    PuctNodeRequest req(new_node);
    this->scheduler->evaluate(&req);
    this->stats.num_evaluations++;

    return new_node;
}

PuctNode* PuctEvaluator::expandChild(PuctNode* parent, PuctNodeChild* child) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    child->to_node = this->lookupNode(this->basestate_expand_node);
    if (child->to_node != nullptr) {
        this->stats.num_transpositions_attached++;

    } else {
        // create node
        child->unselectable = true;
        parent->unselectable_count++;
        child->to_node = this->createNode(parent, this->basestate_expand_node);
        parent->unselectable_count--;
        child->unselectable = false;
    }

    return child->to_node;
}

///////////////////////////////////////////////////////////////////////////////
// selection - move to new class, should have no side effects.  in/out path:

void PuctEvaluator::setPuctConstant(PuctNode* node, int depth) const {
    const float node_score = node->getCurrentScore(node->lead_role_index);

    // note we have dropped concept of before
    if (node->visits < 8) {
        node->puct_constant = this->conf->puct_constant_init;
        return;
    }

    // get top traversals out from this node
    float top_traversals = 0;
    float total_traversals = 0;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
        if ((float) c->traversals > top_traversals) {
            top_traversals = c->traversals;
        }

        total_traversals += c->traversals;
    }

    if (total_traversals > node->num_children) {
        float ratio = top_traversals / float(total_traversals);

        if (node_score > 0.8 && ratio < 0.95) {
            // let it run away?
            node->puct_constant -= 0.01;

        } else if (node_score < 0.8 && ratio > 0.8) {
            node->puct_constant += 0.01;

        } else if (ratio < 0.55) {
            node->puct_constant -= 0.01;
        }

        // constraints
        // we check score, since if we hit rounding issues, can't figure out anything
        if (depth == 0) {
            node->puct_constant = std::min(node->puct_constant, this->conf->puct_constant_max_root);
            node->puct_constant = std::max(node->puct_constant, this->conf->puct_constant_min_root);

        } else {
            node->puct_constant = std::min(node->puct_constant, this->conf->puct_constant_max);
            node->puct_constant = std::max(node->puct_constant, this->conf->puct_constant_min);
        }
    }
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


PuctNodeChild* PuctEvaluator::selectChild(PuctNode* node, Path& path) {
    ASSERT(!node->isTerminal());
    ASSERT(node->num_children > 0);

    const int depth = path.size();

    // dynamically set the PUCT constant
    this->setPuctConstant(node, depth);

    // nothing to select
    if (node->num_children == 1) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), 0);
        path.emplace_back(node, child, child, node->num_children_expanded);
        return child;
    }

    const float node_score = node->getCurrentScore(node->lead_role_index);

    // prior... (alpha go zero said 0 but there score ranges from [-1,1])
    // original value from network / or terminal value
    float prior_score = node->getFinalScore(node->lead_role_index, true);

    int total_traversals = 0;
    if (this->conf->fpu_prior_discount > 0) {
        float total_policy_visited = 0.0;
        for (int ii=0; ii<node->num_children; ii++) {
            PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
            if (c->to_node != nullptr) {
                if (c->traversals > 0) {
                    total_traversals += c->traversals;
                    total_policy_visited += c->policy_prob;
                }
            }
        }

        float fpu_reduction = this->conf->fpu_prior_discount * std::sqrt(total_policy_visited);
        prior_score -= fpu_reduction;
    }

    const float sqrt_node_visits = std::sqrt(node->visits + 1);

    // get best
    float best_score = -1;
    PuctNodeChild* best_child = nullptr;

    float best_child_score_actual_score = -1;;
    PuctNodeChild* best_child_score = nullptr;

    PuctNodeChild* bad_fallback = nullptr;
    PuctNodeChild* best_fallback = nullptr;

    bool allow_expansions = true;
    if (depth > 0) {
        if (node->visits < this->conf->expand_threshold_visits ||
            (node_score < 0.02 || node_score > 0.98)) {
            // count non final expansions
            int non_final_expansions = 0;
            for (int ii=0; ii<node->num_children; ii++) {
                PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
                if (c->to_node != nullptr && !c->to_node->is_finalised &&
                    (c->to_node->getCurrentScore(node->lead_role_index) > 0.98 ||
                     c->to_node->getCurrentScore(node->lead_role_index) < 0.02)) {
                    non_final_expansions++;
                }
            }

            if (non_final_expansions >= this->conf->number_of_expansions_end_game) {
                allow_expansions = false;
            }
        }
    }

    int unselectables = 0;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);

        // skip unselectables
        if (c->unselectable) {
            unselectables++;
            continue;

        } else if (c->to_node != nullptr &&
                   (c->to_node->num_children > 0 &&
                    c->to_node->unselectable_count == c->to_node->num_children)) {
            unselectables++;
            continue;
        }

        if (c->to_node == nullptr && !allow_expansions) {
            continue;
        }

        // anti latching behaviour (XXX why i need this at all?)
        if (depth == 0 && c->traversals > total_traversals * 0.75) {
            best_fallback = c;
            continue;
        }

       // we use doubles throughtout, for more precision
        double child_score = prior_score;
        double child_pct = c->policy_prob;
        if (this->conf->policy_dilution_visits > 0) {
            const double xprob = 1.0 / (double) node->num_children;
            const double coef = std::min(1.0, this->conf->policy_dilution_visits / (double) node->visits);
            child_pct = coef * c->policy_prob + (1.0 - coef) * xprob;
        }

        if (c->to_node != nullptr) {
            PuctNode* cn = c->to_node;
            child_score = cn->getCurrentScore(node->lead_role_index);

            // ensure finalised are enforced more than other nodes (network can return 1.0f for
            // basically dumb moves, if it thinks it will win regardless)
            if (cn->is_finalised) {
                if (child_score > 0.99) {
                    if (depth > 0) {
                        path.emplace_back(node, c, c, node->num_children_expanded);
                        return c;
                    }

                    child_score *= 1.0f + node->puct_constant;

                } else if (child_score < 0.01) {
                    // ignore this unless no other option
                    bad_fallback = c;
                    continue;

                } else {
                    // no more exploration for you
                    child_pct = 0.0;
                }
            }

            // store the best child
            if (child_score > best_child_score_actual_score) {
                best_child_score_actual_score = child_score;
                best_child_score = c;
            }
        }

        // add inflight_visits to exploration score
        const double inflight_visits = c->to_node != nullptr ? c->to_node->inflight_visits : 0;
        const double denominator = c->traversals + inflight_visits + 1;

        const double exploration_score = node->puct_constant * child_pct * (sqrt_node_visits / denominator);

        // (more exploration) apply score discount for massive number of inflight visits
        // XXX rng - kind of expensive here?  tried using 1 and 0.25... has quite an effect on exploration.
        const double discounted_visits = inflight_visits * (this->rng.get() + 0.25);
        if (c->traversals > 8 && discounted_visits > 0.1) {
            child_score = (child_score * c->traversals) / (c->traversals + discounted_visits);
        }

        // end product
        const double score = child_score + exploration_score;

        if (score > best_score) {
            best_child = c;
            best_score = score;
        }
    }

    // this only happens if there was nothing to select
    if (best_child == nullptr) {
        if (best_fallback != nullptr) {
            best_child = best_fallback;
        } else if (bad_fallback != nullptr) {
            // this is bad, very bad.  There could be a race condition where this keeps getting called
            // ... so we insert a yield just in case.
            if (unselectables > 0) {
                this->scheduler->yield();
            }

            best_child = bad_fallback;

        } else {
            this->stats.num_blocked++;
        }
    }

    if (best_child_score == nullptr) {
        best_child_score = best_child;
    }

    if (best_child != nullptr) {
        path.emplace_back(node, best_child, best_child_score, node->num_children_expanded);
    }

    return best_child;
}


void PuctEvaluator::backUpMiniMax(float* new_scores, const PathElement& cur) {

    // best advice is not to mess with this.  Or of doing it would be best to calcualte in select
    // how much exploration it is.  Then based on how far off the exploration is then back prop
    // that.  Doing anything more complicated here doesn't make any sense.

    // Not that currently we store select debug on the node, we need to store those the path.
    // Especially if going down the multiple select/backprops/

    // valid and enabled?
    if (cur.best == nullptr ||
        cur.best->to_node == nullptr ||
        this->conf->minimax_backup_ratio < 0.0) {
        return;
    }

    // enough visits when children all expanded?
    if (cur.num_children_expanded == cur.node->num_children &&
        cur.node->visits < this->conf->minimax_required_visits) {
        return;
    }

    // was a good choice?
    if (cur.choice == cur.best) {
        return;
    }

    PuctNode* best = cur.best->to_node;
    const int role_index = cur.node->lead_role_index;

    if (role_index != -1) {
        float best_score = best->getCurrentScore(role_index);

        // always the optimist
        bool improving = new_scores[role_index] > best_score;

        // if it looks like it improving, then let it go, i mean, I am all for
        // exploration - but does it have to mess up the tree in the process?

        if (!improving) {
            for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
                double r = this->conf->minimax_backup_ratio;
                new_scores[ii] = (r * best->getCurrentScore(ii) +
                                  (1 - r) * new_scores[ii]);
            }
        }
    }
}


void PuctEvaluator::backPropagate(float* new_scores, const Path& path) {
    const int role_count = this->sm->getRoleCount();
    const int start_index = path.size() - 1;

    bool bp_finalised_only_once = true;

    const PathElement* prev = nullptr;

    for (int index=start_index; index >= 0; index--) {
        const PathElement& cur = path[index];

        ASSERT(cur.node != nullptr);

        if (bp_finalised_only_once &&
            !cur.node->is_finalised && cur.node->lead_role_index >= 0) {
            bp_finalised_only_once = false;

            const PuctNodeChild* best = nullptr;
            {
                float best_score = -1;
                bool more_to_explore = false;
                for (int ii=0; ii<cur.node->num_children; ii++) {
                    const PuctNodeChild* c = cur.node->getNodeChild(role_count, ii);

                    if (c->to_node != nullptr && c->to_node->is_finalised) {
                        float score = c->to_node->getCurrentScore(cur.node->lead_role_index);
                        if (score > best_score) {
                            best_score = score;
                            best = c;
                        }

                    } else {
                        // not finalised, so more to explore
                        more_to_explore = true;
                    }
                }

                // special opportunist case...
                if (best_score > 0.99) {
                    more_to_explore = false;
                }

                if (more_to_explore) {
                    best = nullptr;
                }
            }

            if (best != nullptr) {
                for (int ii=0; ii<role_count; ii++) {
                    cur.node->setCurrentScore(ii, best->to_node->getCurrentScore(ii));
                }

                cur.node->is_finalised = true;
            }
        }

        if (cur.node->is_finalised) {
            // This is important.  If we are backpropagating some path which is exploring, the
            // finalised scores take precedent
            // Also important for transpositions (if ever implemented)
            for (int ii=0; ii<role_count; ii++) {
                new_scores[ii] = cur.node->getCurrentScore(ii);
            }

        } else {
            // if configured, will minimax
            this->backUpMiniMax(new_scores, cur);

            float scaled_visits = cur.node->visits;

            const float scaled_anchor = this->conf->scaled_visits_at;
            if (scaled_anchor > 0 && cur.node->visits > scaled_anchor) {

                const bool parent_is_final = prev != nullptr && prev->node->is_finalised;
                const float scale = parent_is_final ? this->conf->scaled_visits_finalised_reduce : this->conf->scaled_visits_reduce;

                const float leftover = cur.node->visits - scaled_anchor;

                scaled_visits = scaled_anchor + leftover / scale;
            }

            for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
                float score = ((scaled_visits * cur.node->getCurrentScore(ii) + new_scores[ii]) /
                               (scaled_visits + 1.0));

                cur.node->setCurrentScore(ii, score);
            }
        }

        cur.node->visits++;

        if (cur.node->inflight_visits > 0) {
            cur.node->inflight_visits--;
        }

        if (cur.choice != nullptr) {
            cur.choice->traversals++;
        }

        prev = &cur;
    }
}

int PuctEvaluator::treePlayout() {

    PuctNode* current = this->root;
    ASSERT(current != nullptr && !current->isTerminal());

    std::vector <PathElement> path;
    float scores[this->sm->getRoleCount()];

    PuctNodeChild* child = nullptr;

    while (true) {
        ASSERT(current != nullptr);

        // End of the road
        // XXX this needs to be different if self play
        if (current->is_finalised) {
            path.emplace_back(current, nullptr, nullptr, current->num_children_expanded);
            break;
        }

        // Choose selection
        while (true) {
            child = this->selectChild(current, path);
            if (child != nullptr) {
                break;
            }

            this->scheduler->yield();
        }

        // if does not exist, then create it (will incur a nn prediction)
        if (child->to_node == nullptr) {
            current = this->expandChild(current, child);

            // end of the road.  We don't continue if num_children == 1, since there is nothing to
            // select
            if (current->is_finalised || current->num_children > 1) {
                // why do we add this?  There is no visits! XXX
                path.emplace_back(current, nullptr, nullptr, current->num_children_expanded);
                break;
            }

            // continue
        }

        current->inflight_visits++;
        current = child->to_node;
    }

    if (current->is_finalised) {
        this->stats.playouts_finals++;
    }

    for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
        scores[ii] = current->getCurrentScore(ii);
    }

    this->backPropagate(scores, path);

    this->stats.num_tree_playouts++;
    return path.size();
}

void PuctEvaluator::normaliseRootNode() {
    if (this->conf->root_node_normalisation_limit < 0) {
        return;
    }

    ASSERT(this->conf->root_node_normalisation_limit > 0);

    if (this->root == nullptr || (int) this->root->visits <= this->conf->root_node_normalisation_limit) {
        return;
    }

    double scale = this->conf->root_node_normalisation_limit / (double) this->root->visits;

    // not worth it, cause more rounding issues than anything else
    if (scale > 0.95) {
        return;
    }


    uint32_t total_reduction = 0;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);

        uint32_t new_traversals = (c->traversals * scale) + 1;
        if (new_traversals < c->traversals) {
            total_reduction += c->traversals - new_traversals;
            c->traversals = new_traversals;
        }
    }

    K273::l_info("Reducing traversals on root by %d", total_reduction);
    ASSERT(this->root->visits > total_reduction);
    this->root->visits -= total_reduction;
}

void PuctEvaluator::playoutWorker() {
    // loops until done
    while (this->do_playouts) {
        if (this->root->is_finalised) {
            break;
        }

        int depth = this->treePlayout();
        this->stats.playouts_max_depth = std::max(depth, this->stats.playouts_max_depth);
        this->stats.playouts_total_depth += depth;
    }
}

void PuctEvaluator::playoutMain(double end_time) {
    const double start_time = K273::get_time();
    K273::l_debug("enter playoutMain() for max %.1f seconds", end_time - start_time);

    const bool use_think_time = this->conf->think_time > 0;

    auto elapsed = [this, start_time](double multiplier) {
        return K273::get_time() > (start_time + this->conf->think_time * multiplier);
    };

    double next_report_time = K273::get_time() + 2.5;
    auto do_report = [this, &next_report_time]() {
        if (!this->conf->verbose) {
            return false;
        }

        double t = K273::get_time();
        if (t > next_report_time) {
            next_report_time = t + 2.5;
            return true;
        }

        return false;
    };

    int iterations = 0;
    while (true) {
        const int our_role_index = this->root->lead_role_index;

        if (this->root->is_finalised && iterations > 1000) {
            K273::l_warning("Breaking early as finalised");
            break;
        }

        if (end_time > 0 && K273::get_time() > end_time) {
            K273::l_warning("Hit hard time limit");
            break;
        }

        // use think time:
        // hacked up so can run in reasonable times during ICGA
        if (use_think_time && iterations % 20 == 0 && K273::get_time() > (start_time + 2.0)) {
            //const float our_score = this->root->getCurrentScore(our_role_index);
            //if (our_score > 0.99 || our_score < 0.010) {
            //    if (this->converged(2)) {
            //        K273::l_warning("Done done (simple converge)");
            //        break;
            //    }
            //}

            // if (our_score > 0.97 || our_score < 0.03) {
            //     if (elapsed(0.25) && this->converged(this->conf->converge_non_relaxed)) {
            //         K273::l_warning("End game converged (relaxed)");
            //         break;
            //     }

            //     if (elapsed(0.33) && this->converged(2)) {
            //         K273::l_warning("End game done (simple converge)");
            //         break;
            //     }

            //     if (elapsed(0.5)) {
            //         K273::l_warning("End game done (no converge)");
            //         break;
            //     }
            // }

            if (elapsed(1.0) && this->converged(this->conf->converge_relaxed)) {
                K273::l_warning("Breaking since converged (relaxed)");
                break;
            }

            if (elapsed(1.33) && this->converged(this->conf->converge_non_relaxed)) {
                K273::l_warning("Breaking since converged (non-relaxed)");
                break;
            }

            if (elapsed(1.75)) {
                K273::l_warning("Breaking - but never converged :(");
                break;
            }
        }

        // do some work here
        int depth = this->treePlayout();
        this->stats.playouts_max_depth = std::max(depth, this->stats.playouts_max_depth);
        this->stats.playouts_total_depth += depth;

        iterations++;

        if (do_report()) {
            const PuctNodeChild* best = this->chooseTopVisits(this->root);
            if (best->to_node != nullptr) {
                const int choice = best->move.get(our_role_index);
                K273::l_info("Evals %d/%d/%d, depth %.2f/%d, n/t: %d/%d, best: %.4f, move: %s",
                             this->stats.num_evaluations,
                             this->stats.num_tree_playouts,
                             this->stats.playouts_finals,
                             this->stats.playouts_total_depth / float(this->stats.num_tree_playouts),
                             this->stats.playouts_max_depth,
                             this->number_of_nodes,
                             this->stats.num_transpositions_attached,
                             best->to_node->getCurrentScore(our_role_index),
                             this->sm->legalToMove(our_role_index, choice));
            }
        }
    }

    if (this->conf->verbose) {
        if (this->stats.num_tree_playouts) {
            K273::l_info("Time taken for %d evaluations in %.3f seconds",
                         this->stats.num_evaluations, K273::get_time() - start_time);

            K273::l_debug("The average depth explored: %.2f, max depth: %d",
                          this->stats.playouts_total_depth / float(this->stats.num_tree_playouts),
                          this->stats.playouts_max_depth);
        } else {
            K273::l_debug("Did no tree playouts.");
        }

        if (this->stats.num_blocked) {
            K273::l_warning("Number of blockages %d", this->stats.num_blocked);
        }
    }
}

PuctNode* PuctEvaluator::fastApplyMove(const PuctNodeChild* next) {
    ASSERT(this->root != nullptr);

    const int number_of_nodes_before = this->number_of_nodes;

    PuctNode* new_root = nullptr;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);

        if (c == next) {
            ASSERT(new_root == nullptr);
            if (c->to_node == nullptr) {
                this->expandChild(this->root, c);
            }

            new_root = c->to_node;

        } else {
            if (c->to_node != nullptr) {
                PuctNode* next_node = c->to_node;
                c->to_node = nullptr;

                ASSERT(next_node->ref_count > 0);
                next_node->ref_count--;
                if (next_node->ref_count == 0) {
                    this->releaseNodes(next_node);
                    this->garbage.push_back(next_node);
                }
            }
        }
    }

    if (this->garbage.size()) {
        K273::l_error("Garbage collected... %zu, please wait", this->garbage.size());
        for (PuctNode* n : this->garbage) {
            this->removeNode(n);
        }
    }

    this->garbage.clear();

    ASSERT(new_root != nullptr);

    this->root->ref_count--;
    if (this->root->ref_count == 0) {
        //K273::l_debug("Removing root node");
        this->removeNode(this->root);

    } else {
        K273::l_debug("What is root ref_count? %d", this->root->ref_count);
    }

    this->root = new_root;
    this->game_depth++;

    if (number_of_nodes_before - this->number_of_nodes > 0) {
        K273::l_info("deleted %d nodes", number_of_nodes_before - this->number_of_nodes);
    }

    return this->root;
}

void PuctEvaluator::applyMove(const GGPLib::JointMove* move) {
    // XXX this is only here for the player.  We should probably have a player class, and not
    // simplify code greatly.
    bool found = false;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);
        if (c->move.equals(move)) {
            this->fastApplyMove(c);
            found = true;
            break;
        }
    }

    std::string move_str = PuctNode::moveString(*move, this->sm);

    if (!found) {
        K273::l_warning("PuctEvaluator::applyMove(): Did not find move %s",
                        move_str.c_str());
    } else {
        K273::l_info("PuctEvaluator::applyMove(): %s", move_str.c_str());
    }

    ASSERT(this->root != nullptr);
}

void PuctEvaluator::reset(int game_depth) {
    // really free all
    if (this->root != nullptr) {
        this->releaseNodes(this->root);
        this->garbage.push_back(this->root);

        K273::l_error("Garbage collected... %zu, please wait", this->garbage.size());
        for (PuctNode* n : this->garbage) {
            this->removeNode(n);
        }

        this->garbage.clear();

        this->root = nullptr;
    }

    this->stats.reset();

    if (this->number_of_nodes) {
        K273::l_warning("Number of nodes not zero %d", this->number_of_nodes);
    }

    if (this->node_allocated_memory) {
        K273::l_warning("Leaked memory %ld", this->node_allocated_memory);
    }

    // this is the only place we set game_depth
    this->game_depth = game_depth;
}

PuctNode* PuctEvaluator::establishRoot(const GGPLib::BaseState* current_state) {
    ASSERT(this->root == nullptr);

    if (current_state == nullptr) {
        current_state = this->sm->getInitialState();
    }

    this->sm->updateBases(current_state);
    this->root = this->createNode(nullptr, current_state);
    this->root->game_depth = this->game_depth;

    ASSERT(!this->root->isTerminal());
    return this->root;
}

const PuctNodeChild* PuctEvaluator::onNextMove(int max_evaluations, double end_time) {
    ASSERT(this->root != nullptr);

    this->stats.reset();
    this->do_playouts = true;

    this->normaliseRootNode();

    // this will be spawned as a coroutine (see addRunnable() below)
    int worker_count = 0;
    auto f = [this, &worker_count]() {
        this->playoutWorker();
        worker_count--;
    };

    if (this->conf->batch_size > 1) {

        if (this->root != nullptr && !this->root->is_finalised) {

            if (max_evaluations < 0 || max_evaluations > 1000) {
                for (int ii=0; ii<this->conf->batch_size - 1; ii++) {
                    worker_count++;
                    this->scheduler->addRunnable(f);
                }
            }
        }
    }

    if (max_evaluations != 0) {
        this->playoutMain(end_time);
    }

    // collect workers
    this->do_playouts = false;
    while (worker_count > 0) {
        this->scheduler->yield();
    }

    K273::l_verbose("All workers collected.");

    const PuctNodeChild* choice = this->choose();

    // this is a hack to only show tree when it is our 'turn'.  Be better to use bypass opponent turn
    // flag than abuse this value (XXX).
    if (max_evaluations != 0 && this->conf->verbose) {
        this->logDebug(choice);
    }

    return choice;
}

float PuctEvaluator::getTemperature() const {
    if (this->game_depth >= this->conf->depth_temperature_stop) {
        return -1;
    }

    ASSERT(this->conf->temperature > 0);

    float multiplier = 1.0f + ((this->game_depth - this->conf->depth_temperature_start) *
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

const PuctNodeChild* PuctEvaluator::chooseTopVisits(const PuctNode* node) {
    if (node == nullptr) {
        node = this->root;
    }

    if (node == nullptr) {
        return nullptr;
    }

    const int role_index = node->lead_role_index;

    auto children = PuctNode::sortedChildrenTraversals(node, this->sm->getRoleCount());

    // look for finalised first
    if (node->is_finalised && node->getCurrentScore(role_index) > 1.0) {
        for (auto c : children) {
            if (c->to_node != nullptr && c->to_node->is_finalised &&
                c->to_node->getCurrentScore(role_index) > 1.0) {
                return c;
            }
        }
    }

    // compare top two.  This is a heuristic to cheaply check if the node hasn't yet converged and
    // chooses the one with the best score.  It isn't very accurate, the only way to get 100%
    // accuracy is to keep running for longer, until it cleanly converges.  This is a best guess for now.
    if (this->conf->top_visits_best_guess_converge_ratio > 0 && children.size() >= 2) {
        PuctNode* n0 = children[0]->to_node;
        PuctNode* n1 = children[1]->to_node;

        if (n0 != nullptr && n1 != nullptr) {
            if (children[1]->traversals > children[0]->traversals * this->conf->top_visits_best_guess_converge_ratio &&
                n1->getCurrentScore(role_index) > n0->getCurrentScore(role_index)) {
                return children[1];
            } else {
                return children[0];
            }
        }
    }

    ASSERT(children.size() > 0);
    return children[0];
}

const PuctNodeChild* PuctEvaluator::chooseTemperature(const PuctNode* node) {
    if (node == nullptr) {
        node = this->root;
    }

    float temperature = this->getTemperature();
    if (temperature < 0) {
        return this->chooseTopVisits(node);
    }

    // subtle: when the visits is low (like 0), we want to use the policy part of the
    // distribution. By using linger here, we get that behaviour.
    Children dist;
    if (root->visits < root->num_children) {
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

Children PuctEvaluator::getProbabilities(PuctNode* node, float temperature, bool use_linger) {
    // XXX this makes the assumption that our legals are unique for each child.

    ASSERT(node->num_children > 0);

    // since we add 0.1 to each our children (this is so the percentage does don't drop too low)
    float node_visits = node->visits + 0.1 * node->num_children;

    // add some smoothness.  This also works for the case when doing no evaluations (ie
    // onNextMove(0)), as the node_visits == 0 and be uniform.
    float linger_pct = 0.1f;

    float total_probability = 0.0f;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        int child_visits = child->to_node ? child->traversals + 0.1f : 0.1f;
        if (use_linger) {
            child->next_prob = linger_pct * child->policy_prob + (1 - linger_pct) * (child_visits / node_visits);

        } else {
            child->next_prob = child_visits / node_visits;
        }

        // apply temperature
        child->next_prob = ::pow(child->next_prob, temperature);
        total_probability += child->next_prob;
    }

    // normalise it
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        child->next_prob /= total_probability;
    }

    return PuctNode::sortedChildren(node, this->sm->getRoleCount(), true);
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
            if (cur->visits < cur->num_children) {
                dist = this->getProbabilities(cur, 1.2, true);
            } else {
                dist = this->getProbabilities(cur, 1.2, false);
            }
        }

        PuctNode::dumpNode(cur, next_choice, indent, sort_by_next_probability, this->sm);

        if (next_choice == nullptr || next_choice->to_node == nullptr) {
            break;
        }

        cur = next_choice->to_node;
    }
}
