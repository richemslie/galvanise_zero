
#include "puct/evaluator.h"
#include "puct/node.h"

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
#include <algorithm>
#include <unordered_set>

using namespace GGPZero;

///////////////////////////////////////////////////////////////////////////////

PathElement::PathElement(PuctNode* node, PuctNodeChild* choice,
                         PuctNodeChild* best) :
    node(node),
    choice(choice),
    best(best) {
}

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::PuctEvaluator(GGPLib::StateMachineInterface* sm, NetworkScheduler* scheduler,
                             const GGPZero::GdlBasesTransformer* transformer) :
    conf(nullptr),
    sm(sm),
    basestate_expand_node(nullptr),
    scheduler(scheduler),
    game_depth(0),
    initial_root(nullptr),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0),
    do_playouts(false) {

    this->basestate_expand_node = this->sm->newBaseState();

    this->lookup = GGPLib::BaseState::makeMaskedMap <PuctNode*>(transformer->createHashMask(this->sm->newBaseState()));
}

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::~PuctEvaluator() {
    free(this->basestate_expand_node);
    this->reset(0);
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::updateConf(const PuctConfig* conf) {
    if (conf->verbose) {
        K273::l_verbose("config verbose: %d, dump_depth: %d, choice: %s", conf->verbose,
                        conf->max_dump_depth,
                        (conf->choose == ChooseFn::choose_top_visits) ? "choose_top_visits"
                                                                      : "choose_temperature");

        K273::l_verbose("think %.1f, puct constant %.2f, root: %.2f, batch_size=%d",
                        conf->think_time, conf->puct_constant, conf->puct_constant_root,
                        conf->batch_size);

        K273::l_verbose("dirichlet_noise (pct: %.2f), fpu_prior_discount: %.2f/%.2f",
                        conf->dirichlet_noise_pct, conf->fpu_prior_discount,
                        conf->fpu_prior_discount_root);

        K273::l_verbose("noise policy squash (pct: %.2f, prob: %.2f),",
                        conf->noise_policy_squash_pct, conf->noise_policy_squash_prob);

        K273::l_verbose(
            "temperature: %.2f, start(%d), stop(%d), incr(%.2f), max(%.2f) scale(%.2f)",
            conf->temperature, conf->depth_temperature_start, conf->depth_temperature_stop,
            conf->depth_temperature_max, conf->depth_temperature_increment, conf->random_scale);

        K273::l_verbose(
            "converge: visits=%d, multiplier=%1f, ratio=%.2f, ",
            conf->converged_visits, conf->evaluation_multiplier_to_convergence,
            conf->top_visits_best_guess_converge_ratio);

        K273::l_verbose("transpositions %s / backup finalised %s, use_legals_count_draw %d",
                        conf->lookup_transpositions ? "true" : "false",
                        conf->backup_finalised ? "true" : "false", conf->use_legals_count_draw);
    }

    this->conf = conf;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::removeNode(PuctNode* node) {
    const GGPLib::BaseState* bs = node->getBaseState();
    if (this->conf->lookup_transpositions) {
        this->lookup->erase(bs);
    }

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
            if (next_node->ref_count <= 0) {
                K273::l_warning("A cycle was found in Player::releaseNodes() skipping %d",
                                next_node->ref_count);
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

PuctNode* PuctEvaluator::lookupNode(const GGPLib::BaseState* bs, int depth) {
    if (!this->conf->lookup_transpositions) {
        return nullptr;
    }

    auto found = this->lookup->find(bs);
    if (found != this->lookup->end()) {
        PuctNode* result = found->second;

        // this is generally bad, as it may end up in a cycle... so no transposition in this case
        if (result->game_depth != depth) {
            //K273::l_warning("Lookup may form a cycle - skipping");
            return nullptr;
        }

        return result;
    }

    return nullptr;
}

PuctNode* PuctEvaluator::createNode(PuctNode* parent, const GGPLib::BaseState* state) {

    // constraint sm, already set
    PuctNode* new_node = PuctNode::create(state, this->sm);

    // add to lookup table
    if (this->conf->lookup_transpositions) {
        this->lookup->emplace(new_node->getBaseState(), new_node);
    }

    // update stats
    this->number_of_nodes++;
    this->node_allocated_memory += new_node->allocated_size;

    new_node->parent = parent;
    if (parent != nullptr) {
        new_node->game_depth = parent->game_depth + 1;
        parent->num_children_expanded++;

    } else {
        new_node->game_depth = this->game_depth;
    }

    if (new_node->is_finalised) {
        // hack to try and focus more on winning lines
        // (XXX) actually a very good hack... maybe make it less hacky somehow
        for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
            const float s = new_node->getCurrentScore(ii);
            if (s > 0.99) {
                new_node->setCurrentScore(ii, s * 1.05);
            } else if (s < 0.01) {
                new_node->setCurrentScore(ii, -0.05);
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

    int next_depth = parent->game_depth + 1;
    child->to_node = this->lookupNode(this->basestate_expand_node, next_depth);

    if (child->to_node != nullptr) {
        child->to_node->ref_count++;
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


typedef std::vector <PuctNodeChild*> SortedChildren;
static SortedChildren sortedChildrenSelect(PuctNode* node, int role_count) {

    SortedChildren children;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(role_count, ii);
        children.push_back(child);
    }

    auto f = [node](const PuctNodeChild* a, const PuctNodeChild* b) {
        float sa = a->to_node == nullptr ? -1 : a->to_node->getCurrentScore(node->lead_role_index);
        float sb = b->to_node == nullptr ? -1 : b->to_node->getCurrentScore(node->lead_role_index);

        if (sa < 0 && sb < 0) {
            return a->policy_prob_orig > b->policy_prob_orig;
        }

        return sa > sb;
    };

    std::sort(children.begin(), children.end(), f);
    return children;
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
        path.emplace_back(node, child, child);
        return child;
    }

    // XXX add an option to turn of dirichlet noise at depth 1
    if (depth == 0) {
        this->setDirichletNoise(node);
    }

    float prior_score = this->priorScore(node, depth);
    const double sqrt_node_visits = std::sqrt(node->visits + 1);

    // get best
    float best_score = -1;
    PuctNodeChild* best_child = nullptr;

    float best_child_score_actual_score = -1;
    PuctNodeChild* best_child_score = nullptr;

    PuctNodeChild* bad_fallback = nullptr;

    float best_fallback_score = -1;
    PuctNodeChild* best_fallback = nullptr;

    int unselectables = 0;

    auto children = sortedChildrenSelect(node, this->sm->getRoleCount());

    int count = 0;
    for (PuctNodeChild* c : children) {
        count++;

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

        // we use doubles throughout, for more precision
        double child_score = prior_score;
        const int traversals = c->traversals + 1;

        // add inflight_visits to exploration score
        const double inflight_visits = c->to_node != nullptr ? c->to_node->inflight_visits : 0;

        // standard PUCT as per AG0 paper
        double exploration_score = node->puct_constant * c->policy_prob * sqrt_node_visits / (traversals + inflight_visits);

        if (c->to_node != nullptr) {
            PuctNode* cn = c->to_node;
            child_score = cn->getCurrentScore(node->lead_role_index);

            // ensure finalised are enforced more than other nodes (network can return 1.0f for
            // basically dumb moves, if it thinks it will win regardless)
            if (cn->is_finalised) {
                if (child_score > 0.99) {
                    if (depth > 0) {
                        path.emplace_back(node, c, c);
                        return c;
                    }

                    // XXX do this with 1.05 ??
                    child_score *= 1.0f + node->puct_constant;

                } else if (child_score < 0.01) {
                    // ignore this unless no other option
                    bad_fallback = c;
                    continue;

                } else {
                    // no more exploration for you
                    exploration_score = 0.0;
                }
            }

            // store the best child (only if the number of visits is ok)
            if ((cn->is_finalised || cn->visits > 42) &&
                child_score > best_child_score_actual_score) {
                best_child_score_actual_score = child_score;
                best_child_score = c;
            }
        }

        // (more exploration) apply score discount for massive number of inflight visits
        if (c->traversals > 0 && inflight_visits > 0) {
            const double discounted_visits = inflight_visits * (this->rng.get() + 0.5);
            child_score = (child_score * c->traversals) / (c->traversals + discounted_visits);
        }

        // XXX add this back in *again*.  Basically in words: at the root node, if we are not
        // exploring then there isn't any point in doing any search (this is for cases where it
        // exploits 99% on one child).

        float limit_latch_root = 0.66;

        // end product
        // use for debug/display
        c->debug_node_score = child_score;
        c->debug_puct_score = exploration_score;

        const double score = child_score + exploration_score;

        if (node->visits > 1000 &&
            node->visits < 40000000 &&
            depth == 0 &&
            this->rng.get() > 0.1) {

            if (c->traversals > 16 && c->traversals > node->visits * limit_latch_root) {

                if (best_fallback == nullptr || score > best_fallback_score) {
                    best_fallback = c;
                    best_fallback_score = score;
                }

                continue;
            }
        }

        if (score > best_score) {
            best_child = c;
            best_score = score;
        }
    }

    // this only happens if there was nothing to select
    if (best_child == nullptr) {

        if (best_fallback != nullptr) {
            if (best_child_score != nullptr) {
                best_child = best_child_score;

            } else {
                best_child = best_fallback;
            }

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
        path.emplace_back(node, best_child, best_child_score);
    }

    return best_child;
}

void PuctEvaluator::backup(float* new_scores, const Path& path) {
    const int role_count = this->sm->getRoleCount();

    auto forceFinalise = [role_count](PuctNode* cur) -> const PuctNodeChild* {
        float best_score = -1;
        const PuctNodeChild* best = nullptr;
        bool more_to_explore = false;

        for (int ii=0; ii<cur->num_children; ii++) {
            const PuctNodeChild* c = cur->getNodeChild(role_count, ii);

            if (c->to_node != nullptr && c->to_node->is_finalised) {
                float score = c->to_node->getCurrentScore(cur->lead_role_index);

                // opportunist case
                if (score > 0.99) {
                    return c;
                }

                if (score > best_score) {
                    best_score = score;
                    best = c;
                }

            } else {
                // not finalised, so more to explore...
                more_to_explore = true;
            }
        }

        if (more_to_explore) {
            return nullptr;
        }

        return best;
    };

    bool bp_finalised_only_once = this->conf->backup_finalised;
    const PathElement* prev = nullptr;

    for (int index=path.size() - 1; index >= 0; index--) {
        const PathElement& cur = path[index];

        ASSERT(cur.node != nullptr);

        if (bp_finalised_only_once &&
            !cur.node->is_finalised && cur.node->lead_role_index >= 0) {
            bp_finalised_only_once = false;

            const PuctNodeChild* finalised_child = forceFinalise(cur.node);
            if (finalised_child != nullptr) {
                for (int ii=0; ii<role_count; ii++) {
                    cur.node->setCurrentScore(ii, finalised_child->to_node->getCurrentScore(ii));
                }

                cur.node->is_finalised = true;
            }
        }

        if (cur.node->is_finalised) {
            // This is important.  If we are backpropagating some path which is exploring, the
            // finalised scores take precedent.  Also important for transpositions.
            for (int ii=0; ii<role_count; ii++) {
                new_scores[ii] = cur.node->getCurrentScore(ii);
            }

        } else {
            for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
                float visits = cur.node->visits;
                if (visits > 100000) {
                    visits = 100000 + 0.1f * (visits - 100000);
                }

                float score = ((visits * cur.node->getCurrentScore(ii) + new_scores[ii]) /
                               (visits + 1.0f));

                cur.node->setCurrentScore(ii, score);
            }
        }

        cur.node->visits++;

        if (cur.node->inflight_visits > 0) {
            cur.node->inflight_visits--;
        }

        if (cur.choice != nullptr) {
            cur.choice->traversals++;

            // apply policy dilution (XXX add an option)
            if (cur.node->visits > 23) {
                const float cur_score = cur.node->getCurrentScore(cur.node->lead_role_index);

                // widen ranges
                if (cur_score > 0.3 && cur_score < 0.7) {

                    // reasonable?
                    const float policy_dilute_apply = 0.995;
                    const float policy_dilute_min = 0.02f;

                    if (cur.choice->policy_prob > policy_dilute_min) {
                        cur.choice->policy_prob *= policy_dilute_apply;
                        cur.choice->policy_prob = std::max(policy_dilute_min, cur.choice->policy_prob);
                    }

                } else if (cur_score > 0.15 && cur_score < 0.85) {

                    // reasonable?
                    const float policy_dilute_apply = 0.9975;
                    const float policy_dilute_min = 0.03f;

                    if (cur.choice->policy_prob > policy_dilute_min) {
                        cur.choice->policy_prob *= policy_dilute_apply;
                        cur.choice->policy_prob = std::max(policy_dilute_min, cur.choice->policy_prob);
                    }

                } else {

                    // reasonable?
                    const float policy_dilute_apply = 0.9975;
                    const float policy_dilute_min = 0.10f;

                    if (cur.choice->policy_prob > policy_dilute_min) {
                        cur.choice->policy_prob *= policy_dilute_apply;
                        cur.choice->policy_prob = std::max(policy_dilute_min, cur.choice->policy_prob);
                    }
                }
            }
        }

        if (cur.node->visits % 100 == 0) {
            // normalise policy_prob
            cur.node->normaliseX();
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

        if (current->isTerminal()) {
            path.emplace_back(current, nullptr, nullptr);
            break;
        }

        if (current->is_finalised) {
            path.emplace_back(current, nullptr, nullptr);
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

            // end of the road.  We continue if num_children == 1, since there is nothing to
            // select
            if (current->is_finalised || current->num_children > 1) {
                // why do we add this?  There is no visits! XXX
                path.emplace_back(current, nullptr, nullptr);
                break;
            }
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

    this->backup(scores, path);

    this->stats.num_tree_playouts++;
    return path.size();
}

void PuctEvaluator::playoutWorker(int worker_id) {
    while (this->do_playouts) {
        // prevent race condition? XXX
        // theory is if hitting terminal nodes during treePlayout(), but does not manage to
        // finalise root - then this will become a tight loop - and this->do_playouts will never be
        // set from playoutMain()
        if (this->stats.num_tree_playouts % 10000 == 0) {
            this->scheduler->yield();
        }

        if (this->root->is_finalised) {
            break;
        }

        int depth = this->treePlayout();

        this->stats.playouts_max_depth = std::max(depth, this->stats.playouts_max_depth);
        this->stats.playouts_total_depth += depth;
    }
}

void PuctEvaluator::playoutMain(int max_evaluations, double end_time) {
    const double start_time = K273::get_time();
    if (this->conf->verbose) {
        K273::l_debug("enter playoutMain() for max %.1f seconds", end_time - start_time);

        if (this->root->is_finalised) {
            K273::l_warning("In playoutMain() and root is finalised");
        }
    }

    const bool use_think_time = this->conf->think_time > 0;

    auto elapsed = [start_time](double elasped_time) {
        return K273::get_time() > (start_time + elasped_time);
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


#define LOG_BREAK(fmt, ...)                           \
    if (this->conf->verbose) {                        \
        K273::l_warning(fmt, ##__VA_ARGS__);          \
    }                                                 \
    break;                                            \

    const int max_non_converged_evaluations = max_evaluations * conf->evaluation_multiplier_to_convergence;

    // this is different from evaluations, and allows for hitting terminal nodes
    const int max_tree_playouts = 4 * max_non_converged_evaluations;

    while (true) {
        const int our_role_index = this->root->lead_role_index;
        const bool is_converged = this->converged(this->conf->converged_visits);

        // A long list of break conditions:

        // note this is hard time, and is different from think time
        if (end_time > 0 && K273::get_time() > end_time) {
            LOG_BREAK("Hit hard time limit");
        }

        // finalised?
        if (this->root->is_finalised && this->stats.num_tree_playouts > 100) {
            LOG_BREAK("Breaking early as finalised");
        }

        if (is_converged && this->stats.num_tree_playouts > max_tree_playouts) {
            LOG_BREAK("Breaking max tree playouts");
        }

        // XXX add parameter for this
        if (this->number_of_nodes > 50000000) {
            LOG_BREAK("Breaking max nodes");
        }

        if (is_converged && this->stats.num_evaluations > max_evaluations) {
            LOG_BREAK("Breaking max evaluations (converged).");
        }

        if (!is_converged && this->stats.num_evaluations > max_non_converged_evaluations) {
            LOG_BREAK("Breaking max evaluations (non-converged).");
        }

        if (use_think_time) {
            if (is_converged && elapsed(this->conf->think_time)) {
                LOG_BREAK("Breaking (converged) - think time elapsed.");
            }

            if (!is_converged &&
                elapsed(this->conf->think_time * this->conf->evaluation_multiplier_to_convergence)) {
                LOG_BREAK("Breaking (non-converged) - think time elapsed.");
            }

            // break early if game is done (XXX add a parameter for this value)
            if (elapsed(120.0) && is_converged) {

                const PuctNodeChild* best = this->chooseTopVisits(this->root);

                if (best->to_node->getCurrentScore(our_role_index) > 0.975 ||
                    best->to_node->getCurrentScore(our_role_index) < 0.025) {

                    LOG_BREAK("Breaking early. game over, and converged. ");
                }
            }
        }

        // and... do some work here
        int depth = this->treePlayout();

        // update some stats
        this->stats.playouts_max_depth = std::max(depth, this->stats.playouts_max_depth);
        this->stats.playouts_total_depth += depth;

        if (do_report()) {
            const PuctNodeChild* best = this->chooseTopVisits(this->root);
            if (best->to_node != nullptr) {
                const int choice = best->move.get(our_role_index);
                K273::l_info("Evals %d/%d/%d, depth %.2f/%d, n/t: %d/%d, best: %.4f, converged %s:, move: %s",
                             this->stats.num_evaluations,
                             this->stats.num_tree_playouts,
                             this->stats.playouts_finals,
                             this->stats.playouts_total_depth / float(this->stats.num_tree_playouts),
                             this->stats.playouts_max_depth,
                             this->number_of_nodes,
                             this->stats.num_transpositions_attached,
                             best->to_node->getCurrentScore(our_role_index),
                             is_converged ? "yes" : "no",
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
    ASSERT(this->initial_root != nullptr);

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
        if (this->conf->verbose) {
            K273::l_error("Garbage collected... %zu, please wait", this->garbage.size());
        }

        for (PuctNode* n : this->garbage) {
            this->removeNode(n);
        }
    }

    this->garbage.clear();

    ASSERT(new_root != nullptr);

    this->root = new_root;
    this->game_depth++;

    if (this->conf->verbose && number_of_nodes_before - this->number_of_nodes > 0) {
        K273::l_info("deleted %d nodes", number_of_nodes_before - this->number_of_nodes);
    }

    return this->root;
}

void PuctEvaluator::applyMove(const GGPLib::JointMove* move) {
    // XXX this is only here for the player.  We should probably have a player class, and simplify code greatly.
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

    if (this->conf->verbose) {
        if (!found) {
            K273::l_warning("PuctEvaluator::applyMove(): Did not find move %s",
                            move_str.c_str());
        } else {
            K273::l_info("PuctEvaluator::applyMove(): %s", move_str.c_str());
        }
    }

    ASSERT(this->root != nullptr);
}

void PuctEvaluator::reset(int game_depth) {
    // really free all
    if (this->initial_root != nullptr) {
        this->releaseNodes(this->initial_root);
        this->garbage.push_back(this->initial_root);

        if (this->conf->verbose) {
            K273::l_error("Garbage collected... %zu, please wait", this->garbage.size());
        }

        for (PuctNode* n : this->garbage) {
            this->removeNode(n);
        }

        this->garbage.clear();

        this->initial_root = this->root = nullptr;
    }

    ASSERT(this->root == nullptr);

    if (this->number_of_nodes) {
        K273::l_warning("Number of nodes not zero %d", this->number_of_nodes);
    }

    if (this->node_allocated_memory) {
        K273::l_warning("Leaked memory %ld", this->node_allocated_memory);
    }

    this->stats.reset();

    // this is the only place we set game_depth
    this->game_depth = game_depth;
}

PuctNode* PuctEvaluator::establishRoot(const GGPLib::BaseState* current_state) {
    ASSERT(this->root == nullptr && this->initial_root == nullptr);

    if (current_state == nullptr) {
        current_state = this->sm->getInitialState();
    }

    this->initial_root = this->root = this->createNode(nullptr, current_state);

    ASSERT(!this->root->isTerminal());
    return this->root;
}

void PuctEvaluator::resetRootNode() {
    ASSERT(this->root != nullptr);

    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);
        // reset policy_prob
        c->policy_prob = c->policy_prob_orig;
        c->traversals = std::min(1U, c->traversals);
    }

    this->root->dirichlet_noise_set = false;
}

const PuctNodeChild* PuctEvaluator::onNextMove(int max_evaluations, double end_time) {
    ASSERT(this->root != nullptr && this->initial_root != nullptr);

    this->stats.reset();
    this->do_playouts = true;

    // automatically reset root node?  XXX a bunch of arbirtary checks to see if will reset it...
    if (conf->think_time > 10 && !this->root->dirichlet_noise_set &&
        !this->root->is_finalised && this->root->visits > 10000) {

        if (this->conf->verbose) {
            K273::l_warning("Warning - reseting root node");
        }

        if (this->number_of_nodes < 3000000) {
            this->resetRootNode();
        }
    }

    // this will be spawned as a coroutine (see addRunnable() below)
    int worker_count = 0;
    auto f = [this, &worker_count]() {
        this->playoutWorker(worker_count);
        worker_count--;
    };

    if (this->conf->batch_size > 1) {

        if (this->root != nullptr && !this->root->is_finalised) {

            if (max_evaluations < 0 || max_evaluations > 100) {
                for (int ii=0; ii<this->conf->batch_size - 1; ii++) {
                    worker_count++;
                    this->scheduler->addRunnable(f);
                }
            }
        }
    }

    if (max_evaluations != 0) {
        this->playoutMain(max_evaluations, end_time);
    }

    // collect workers
    if (this->conf->verbose) {
        K273::l_verbose("Starting collect.");
    }

    this->do_playouts = false;
    while (worker_count > 0) {
        this->scheduler->yield();
    }

    if (this->conf->verbose) {
        K273::l_verbose("All workers collected.");
    }

    const PuctNodeChild* choice = this->choose(this->root);

    // this is a hack to only show tree when it is our 'turn'.  Be better to use bypass opponent turn
    // flag than abuse this value (XXX).
    if (max_evaluations != 0 && this->conf->verbose) {
        this->logDebug(choice);
    }

    return choice;
}

const PuctNodeChild* PuctEvaluator::chooseTopVisits(const PuctNode* node) const {
    ASSERT(node != nullptr);

    auto children = PuctNode::sortedChildrenTraversals(node, this->sm->getRoleCount());
    ASSERT(children.size() > 0);

    const int role_index = node->lead_role_index;

    // look for finalised first
    int indx0 = -1;
    int indx1 = -1;

    int count = 0;
    for (auto c : children) {
        if (c->to_node != nullptr && c->to_node->is_finalised) {

            // finalised / choose forced win move
            if (c->to_node->getCurrentScore(role_index) > 0.99) {
                return c;
            }

            // finalised forced loss / skip
            if (c->to_node->getCurrentScore(role_index) < 0.01) {
                count++;
                continue;
            }
        }

        if (indx0 == -1) {
            indx0 = count;

        } else if (indx1 == -1) {
            indx1 = count;
        }

        count++;
    }

    if (this->conf->top_visits_best_guess_converge_ratio > 0 &&
        indx0 != -1 &&
        indx1 != -1) {

        // compare top two.  This is a heuristic to cheaply check if the node hasn't yet converged and
        // chooses the one with the best score.  It isn't very accurate, the only way to get 100%
        // accuracy is to keep running for longer, until it cleanly converges.  This is a best guess for now.
        const PuctNodeChild* c0 = children[indx0];
        const PuctNodeChild* c1 = children[indx1];

        if (c0->to_node != nullptr && c1->to_node != nullptr) {
            if (c1->traversals > c0->traversals * this->conf->top_visits_best_guess_converge_ratio &&
                c1->to_node->getCurrentScore(role_index) > c0->to_node->getCurrentScore(role_index)) {
                return c1;
            } else {
                return c0;
            }
        }
    }

    return children[0];
}

Children PuctEvaluator::getProbabilities(PuctNode* node, float temperature, bool use_policy) {
    // XXX this makes the assumption that our legals are unique for each child.

    ASSERT(node->num_children > 0);

    // we add 0.001 to each our children, so zero chance doesn't happen
    float node_visits = node->visits + 0.001 * node->num_children;

    float total_probability = 0.0f;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        float child_visits = child->to_node ? child->traversals + 0.001f : 0.001f;
        if (use_policy) {
            child->next_prob = child->policy_prob + 0.001f;

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

    if (node->getCurrentScore(node->lead_role_index) > 0.95) {
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

    bool policy_squash = (this->conf->noise_policy_squash_pct > 0 &&
                          this->rng.get() < this->conf->noise_policy_squash_pct);

    // replace the policy_prob on the node
    float total_policy = 0;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);

        // reduce high fliers
        if (policy_squash) {
            c->policy_prob = std::min(this->conf->noise_policy_squash_prob,
                                      c->policy_prob);
        }

        const float pct = this->conf->dirichlet_noise_pct;
        c->policy_prob = (1.0f - pct) * c->policy_prob + pct * dirichlet_noise[ii];
        total_policy += c->policy_prob;
    }

    // re-normalize node
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

void PuctEvaluator::dumpNode(const PuctNode* node,
                             const PuctNodeChild* choice) const {
    PuctNode::dumpNode(node, choice, "", true, this->sm);
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
