/*

blocked exploration - thinking aloud
====================================

* this is the biggest issue with having massive number of concurrent search/selects.  Without
  backproping the information the tree is being directed into paths it should never go.

  * especiallyin the early phases of the node

  * later on once all nodes have been expanded, then it will still explore more - but much less so

* rather than punt all problems to backprop, what might be an more elegant approach is to expand
  the node with zero visits, and come up with a guess for the score.  Then it can be visited.  The
  score unfortnately is the least of our problems... as their will be no policy until evaluation in
  complete.

  * so is blocking ideal behaviour?  Why visit a node we can't do anything but play randomly from?

  * Since the number of evaluations per turn is going to be large (3-5k per second), and the
    (meaningful) depth of the search is likely to be < 20.  Why are worried about this at all?

  * I think sticking with blocking for now, isn't all that bad.

* given the above, in select we will be doing much more exploration than the formula:

  * blocked nodes (in progress evaluation)
  * scores reduced with virtual visits

* ok, given a carefree attidude to all the above, how best to handle this backprop such that the
  tree isn't being unnessarily swayed.

  * do we care at about extra exploration via virtual visits?  It will work itself in the end.

  * if we know that any children in a node are in progress to be evaluated (TBE) where their policy
    is greater than selected - then we can assume (at selection time) that the node is expanded
    before its time.  What we can do in this case is no backprop whatsoever (essentially zero
    visits).  Is this the best approach?

----

* shmem?  Could get a factor of 4 speed up (maybe).  Still will have the problem of needing even
  more non evaluated nodes.  At least can use multiple GPUs - which will help with turn around
  times of the node evaluations.


simple todos
------------
* add number of final nodes to stats
* add timing stuff to stats
* make a stats struct/class, memset it to zero (or call a method to zero it) rather long list of = 0.
* test the puct variation.  What are good upper and lower bounds.  3.5 still feels high?
  * can we automate this?
  * is it game dependent?  galvanise we got away with one set of numbers.
* (bugix - but inconsequential, I think) fix traversals > visits, when there is only one move

* sort node by policy in node.cpp / PuctNodeRequest::reply()

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

PuctEvaluator::PuctEvaluator(GGPLib::StateMachineInterface* sm, const PuctConfig* conf,
                             NetworkScheduler* scheduler, const GGPZero::GdlBasesTransformer* transformer) :
    sm(sm),
    basestate_expand_node(nullptr),
    conf(conf),
    scheduler(scheduler),
    identifier("PuctEvaluator"),
    game_depth(0),
    evaluations(0),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0) {

    this->basestate_expand_node = this->sm->newBaseState();
    this->updateConf(this->conf);

    // XXX ZZZ tmp
    this->stats_finals = 0;
    this->stats_blocked = 0;
    this->stats_tree_playouts = 0;
    this->stats_transpositions = 0;
    this->stats_total_depth = 0;
    this->stats_max_depth = 0;

    this->lookup = GGPLib::BaseState::makeMaskedMap <PuctNode*>(transformer->createHashMask(this->basestate_expand_node));
}

PuctEvaluator::~PuctEvaluator() {
    free(this->basestate_expand_node);

    this->reset(0);

    delete this->conf;
    delete this->extra;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::updateConf(const PuctConfig* conf, const ExtraPuctConfig* extra_conf) {
    if (extra_conf == nullptr) {
        extra_conf = new ExtraPuctConfig;
    }

    if (conf->verbose) {
        K273::l_verbose("config verbose: %d, dump_depth: %d, puct (%.2f), root_expansions: %d",
                        conf->verbose, conf->max_dump_depth, conf->puct_constant_after,
                        conf->root_expansions_preset_visits);

        K273::l_verbose("dirichlet_noise (alpha: %.2f, pct: %.2f), fpu_prior_discount: %.2f, choose: %s",
                        conf->dirichlet_noise_alpha, conf->dirichlet_noise_pct, conf->fpu_prior_discount,
                        (conf->choose == ChooseFn::choose_top_visits) ? "choose_top_visits" : "choose_temperature");

        K273::l_verbose("temperature: %.2f, start(%d), stop(%d), incr(%.2f), max(%.2f) scale(%.2f)",
                        conf->temperature, conf->depth_temperature_start, conf->depth_temperature_stop,
                        conf->depth_temperature_max, conf->depth_temperature_increment, conf->random_scale);

        K273::l_verbose("Extra!  scaled (at %d, reduce %.2f/%.2f). converge_ratio: %.2f",
                        extra_conf->scaled_visits_at, extra_conf->scaled_visits_reduce,
                        extra_conf->scaled_visits_finalised_reduce,
                        extra_conf->top_visits_best_guess_converge_ratio);
    }

    this->conf = conf;
    this->extra = extra_conf;
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
        return new_node;
    }

    // skips actually evaluation on nodes with only 1 child
    if (new_node->num_children == 1) {
        return new_node;
    }

    // goodbye kansas
    PuctNodeRequest req(new_node);
    this->scheduler->evaluate(&req);
    this->evaluations++;

    return new_node;
}

PuctNode* PuctEvaluator::expandChild(PuctNode* parent, PuctNodeChild* child) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    child->to_node = this->lookupNode(this->basestate_expand_node);
    if (child->to_node != nullptr) {
        this->stats_transpositions++;

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

std::vector <float> PuctEvaluator::getDirichletNoise(int depth) {
    // set dirichlet noise on root?

    std::vector <float> res;

    if (depth != 0) {
        return res;
    }

    if (this->conf->dirichlet_noise_alpha < 0) {
        return res;
    }

    // XXX check with l0 that is still ok.
    std::gamma_distribution <float> gamma(this->conf->dirichlet_noise_alpha, 1.0f);

    float total_noise = 0.0f;
    for (int ii=0; ii<this->root->num_children; ii++) {
        float noise = gamma(this->rng);

        total_noise += noise;
        res.emplace_back(noise);
    }

    // fail if we didn't produce any noise
    if (total_noise < std::numeric_limits<float>::min()) {
        res.clear();
        return res;
    }

    // normalize:
    for (int ii=0; ii<this->root->num_children; ii++) {
       res[ii] /= total_noise;
    }

    return res;
}

void PuctEvaluator::setPuctConstant(PuctNode* node, int depth) const {
    const float node_score = node->getCurrentScore(node->lead_role_index);

    // note we have dropped concept of before
    if (node->visits < 8) {
        node->puct_constant = this->conf->puct_constant_after;
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
        if (depth == 0) {
            node->puct_constant = std::max(node->puct_constant, this->extra->min_puct_root);
        } else {
            node->puct_constant = std::max(node->puct_constant, this->extra->min_puct);
        }

        node->puct_constant = std::min(node->puct_constant, this->extra->max_puct);
    }
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
    }

    // this should be smart enough not to allocate memory unless added to (XXX check)
    std::vector <float> dirichlet_noise = this->getDirichletNoise(depth);
    bool do_dirichlet_noise = !dirichlet_noise.empty();

    const float node_score = node->getCurrentScore(node->lead_role_index);
    if (node_score > 0.95) {
        do_dirichlet_noise = false;
    }

    // prior... (alpha go zero said 0 but there score ranges from [-1,1])
    // original value from network / or terminal value
    float prior_score = node->getFinalScore(node->lead_role_index);
    int total_traversals = 0;
    if (!do_dirichlet_noise && this->conf->fpu_prior_discount > 0) {
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

        float fpu_prior_discount = this->conf->fpu_prior_discount; // + std::max(depth, 10) * 0.01;
        float fpu_reduction = fpu_prior_discount * std::sqrt(total_policy_visited);
        prior_score -= fpu_reduction;
    }

    const float sqrt_node_visits = std::sqrt(node->visits + 1);

    // get best
    float best_score = -1;
    PuctNodeChild* best_child = nullptr;

    float best_child_score_actual_score = -1;;
    PuctNodeChild* best_child_score = nullptr;

    PuctNodeChild* fallback = nullptr;

    bool allow_expansions = true;
    if (node_score < 0.02 || node_score > 0.98) {
        // count non final expansions
        int non_final_expansions = 0;
        for (int ii=0; ii<node->num_children; ii++) {
            PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);
            if (c->to_node != nullptr && !c->to_node->is_finalised) {
                non_final_expansions++;
            }
        }

        // going to assume top 5 moves is enough
        if (non_final_expansions > 5) {
            allow_expansions = false;
        }
    }

    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);

        // skip unselectables
        if (c->unselectable) {
            continue;

        } else if (c->to_node != nullptr &&
                   (c->to_node->num_children > 0 &&
                    c->to_node->unselectable_count == c->to_node->num_children)) {
            continue;
        }

        if (c->to_node == nullptr && !allow_expansions) {
            continue;
        }

        // we use doubles throughtout, for more precision
        double child_score = prior_score;
        double child_pct = c->policy_prob;

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
                    fallback = c;
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

        if (do_dirichlet_noise) {
            double noise_pct = this->conf->dirichlet_noise_pct;
            child_pct = (1.0f - noise_pct) * child_pct + noise_pct * dirichlet_noise[ii];
        }

        // add inflight_visits to exploration score
        const double inflight_visits = c->to_node != nullptr ? c->to_node->inflight_visits : 0;
        const double denominator = c->traversals + inflight_visits + 1;

        const double exploration_score = node->puct_constant * child_pct * (sqrt_node_visits / denominator);

        // (more exploration) apply score discount for massive number of inflight visits
        // XXX rng - kind of expensive here?  tried using 1 and 0.25... has quite an effect on exploration.
        const double discounted_visits = inflight_visits * (this->rng.get() / 2.0);
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
        if (fallback != nullptr) {
            best_child = fallback;
        } else {
            this->stats_blocked++;
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
            float scaled_visits = cur.node->visits;

            const float scaled_anchor = this->extra->scaled_visits_at;
            if (scaled_anchor > 0 && cur.node->visits > scaled_anchor) {

                const bool parent_is_final = prev != nullptr && prev->node->is_finalised;
                const float scale = parent_is_final ? this->extra->scaled_visits_finalised_reduce : this->extra->scaled_visits_reduce;

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
        this->stats_finals++;
    }

    for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
        scores[ii] = current->getCurrentScore(ii);
    }

    this->backPropagate(scores, path);

    this->stats_tree_playouts++;
    return path.size();
}

void PuctEvaluator::playoutLoop(int max_evaluations, double end_time, bool main) {
    double start_time = K273::get_time();
    this->evaluations = 0;

    double next_report_time = K273::get_time() + 2.5;

    int iterations = 0;
    while (true) {
        if (max_evaluations == 0 && this->evaluations > max_evaluations) {
            break;
        }

        if (this->root->is_finalised && iterations > 1000) {
            if (main) {
                K273::l_warning("Breaking early as finalised");
            }

            break;
        }

        if (end_time > 0 && K273::get_time() > end_time) {
            break;
        }

        int depth = this->treePlayout();
        this->stats_max_depth = std::max(depth, this->stats_max_depth);
        this->stats_total_depth += depth;

        if (main && next_report_time > 0 && K273::get_time() > next_report_time) {
            next_report_time = K273::get_time() + 2.5;

            const PuctNodeChild* best = this->chooseTopVisits(this->root);
            if (best->to_node != nullptr) {
                const int our_role_index = this->root->lead_role_index;
                const int choice = best->move.get(our_role_index);
                K273::l_info("Evals %d/%d/%d, depth %.2f/%d, n/t: %d/%d, best: %.4f, move: %s",
                             this->evaluations,
                             this->stats_tree_playouts,
                             this->stats_finals,
                             this->stats_total_depth / float(this->stats_tree_playouts),
                             this->stats_max_depth,
                             this->number_of_nodes,
                             this->stats_transpositions,
                             best->to_node->getCurrentScore(our_role_index),
                             this->sm->legalToMove(our_role_index, choice));
            }
        }

        iterations++;
    }

    if (main && this->conf->verbose) {
        if (this->stats_tree_playouts) {
            K273::l_info("Time taken for %d evaluations in %.3f seconds",
                         this->evaluations, K273::get_time() - start_time);

            K273::l_debug("The average depth explored: %.2f, max depth: %d",
                          this->stats_total_depth / float(this->evaluations),
                          this->stats_max_depth);
        } else {
            K273::l_debug("Did no tree playouts.");
        }


        if (this->stats_blocked) {
            K273::l_warning("Number of blockages %d", this->stats_blocked);
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

    K273::l_error("Garbage collected... %zu, please wait", this->garbage.size());
    for (PuctNode* n : this->garbage) {
        this->removeNode(n);
    }

    this->garbage.clear();

    ASSERT(new_root != nullptr);

    this->root->ref_count--;
    if (this->root->ref_count == 0) {
        K273::l_debug("Removing root node");
        this->removeNode(this->root);

    } else {
        K273::l_debug("What is root ref_count? %d", this->root->ref_count);
    }

    this->root = new_root;
    this->game_depth++;

    K273::l_info("deleted %d nodes", number_of_nodes_before - this->number_of_nodes);

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

    if (!found) {
        K273::l_warning("Did not find move %d / %d", move->get(0), move->get(1));
    }

    ASSERT(this->root != nullptr);
}

void PuctEvaluator::reset(int game_depth) {
    // really free all
    if (this->root != nullptr) {
        this->releaseNodes(this->root);

        K273::l_error("Garbage collected... %zu, please wait", this->garbage.size());
        for (PuctNode* n : this->garbage) {
            this->removeNode(n);
        }

        this->garbage.clear();

        this->root = nullptr;
    }

    this->stats_finals = 0;
    this->stats_blocked = 0;
    this->stats_tree_playouts = 0;
    this->stats_transpositions = 0;
    this->stats_total_depth = 0;
    this->stats_max_depth = 0;

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

    if (this->conf->root_expansions_preset_visits > 0) {
        int number_of_expansions = 0;
        for (int ii=0; ii<this->root->num_children; ii++) {
            PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);

            if (c->to_node == nullptr) {
                this->expandChild(this->root, c);
                c->traversals = std::max(c->traversals,
                                         (uint32_t) this->conf->root_expansions_preset_visits);

                number_of_expansions++;
            }
        }
    }


    auto f = [this, max_evaluations, end_time]() {
        this->playoutLoop(max_evaluations, end_time);
    };

    this->stats_finals = 0;
    this->stats_blocked = 0;
    this->stats_tree_playouts = 0;
    this->stats_transpositions = 0;
    this->stats_total_depth = 0;
    this->stats_max_depth = 0;

    if (max_evaluations == -1 || max_evaluations > 1000) {
        for (int ii=0; ii<31; ii++) {
            this->scheduler->addRunnable(f);
        }
    }

    this->playoutLoop(max_evaluations, end_time, true);

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

    auto children = PuctNode::sortedChildren(node, this->sm->getRoleCount());


    // compare top two.  This is a heuristic to cheaply check if the node hasn't yet converged and
    // chooses the one with the best score.  It isn't very accurate, the only way to get 100%
    // accuracy is to keep running for longer, until it cleanly converges.  This is a best guess for now.
    if (this->extra->top_visits_best_guess_converge_ratio > 0 && children.size() >= 2) {
        PuctNode* n0 = children[0]->to_node;
        PuctNode* n1 = children[1]->to_node;

        if (n0 != nullptr && n1 != nullptr) {
            const int role_index = node->lead_role_index;

            if (children[1]->traversals > children[0]->traversals * this->extra->top_visits_best_guess_converge_ratio &&
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
