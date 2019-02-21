
#include "puct/evaluator.h"
#include "puct/node.h"

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

using namespace GGPZero;

#include "unifify.cpp"

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::PuctEvaluator(GGPLib::StateMachineInterface* sm,
                             const PuctConfig* conf, NetworkScheduler* scheduler,
                             const GGPZero::GdlBasesTransformer* transformer) :
    sm(sm),
    basestate_expand_node(nullptr),
    conf(conf),
    number_repeat_states_draw(-1),
    repeat_states_score(-1.0f),
    scheduler(scheduler),
    game_depth(0),
    evaluations(0),
    initial_root(nullptr),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0) {

    this->basestate_expand_node = this->sm->newBaseState();
    this->masked_bs_equals = new GGPLib::BaseState::EqualsMasked(transformer->createHashMask(this->sm->newBaseState()));

    this->updateConf(this->conf);
}

PuctEvaluator::~PuctEvaluator() {
    free(this->basestate_expand_node);

    this->reset(0);

    delete this->conf;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::updateConf(const PuctConfig* conf) {
    if (conf->verbose) {
        K273::l_verbose("config verbose: %d, max_dump_depth: %d",
                        conf->verbose, conf->max_dump_depth);

        K273::l_verbose("puct_constant: %f, root_expansions_preset_visits: %d",
                        conf->puct_constant, conf->root_expansions_preset_visits);

        K273::l_verbose("dirichlet_noise (alpha: %.2f, pct: %.2f), fpu_prior_discount: %.2f/%.2f",
                        conf->dirichlet_noise_alpha, conf->dirichlet_noise_pct,
                        conf->fpu_prior_discount, conf->fpu_prior_discount_root);

        K273::l_verbose("choose: %s",
                        (conf->choose == ChooseFn::choose_top_visits) ? "choose_top_visits" : "choose_temperature");

        K273::l_verbose("temperature: %.2f, start(%d), stop(%d), incr(%.2f), max(%.2f) scale(%.2f)",
                        conf->temperature, conf->depth_temperature_start, conf->depth_temperature_stop,
                        conf->depth_temperature_max, conf->depth_temperature_increment, conf->random_scale);

        K273::l_verbose("top_visits_best_guess_converge_ratio: %.2f",
                        conf->top_visits_best_guess_converge_ratio);

        K273::l_verbose("evaluation_multipler convergence %.2f",
                        conf->evaluation_multipler_to_convergence);
    }

    this->conf = conf;
}

void PuctEvaluator::setRepeatStateDraw(int number_repeat_states_draw,
                                       float repeat_states_score) {
    this->number_repeat_states_draw = number_repeat_states_draw;
    this->repeat_states_score = repeat_states_score;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::addNode(PuctNode* new_node) {
    this->number_of_nodes++;
    this->node_allocated_memory += new_node->allocated_size;
}

void PuctEvaluator::removeNode(PuctNode* node) {
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        if (child->to_node != nullptr) {
            this->removeNode(child->to_node);
        }

        child->to_node = nullptr;
    }

    this->node_allocated_memory -= node->allocated_size;

    free(node);
    this->number_of_nodes--;
}

void PuctEvaluator::expandChild(PuctNode* parent, PuctNodeChild* child, bool expansion_time) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    // create node
    child->to_node = this->createNode(parent, this->basestate_expand_node, expansion_time);
}

PuctNode* PuctEvaluator::createNode(PuctNode* parent, const GGPLib::BaseState* state, bool expansion_time) {
    PuctNode* new_node = PuctNode::create(state, this->sm);
    if (parent != nullptr) {
        new_node->parent = parent;
        new_node->game_depth = parent->game_depth + 1;
        parent->num_children_expanded++;
    }

    this->addNode(new_node);

    if (new_node->isTerminal()) {
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

    } else {
        if (expansion_time && new_node->num_children == 1) {
            return new_node;
        }

        // goodbye kansas
        PuctNodeRequest req(new_node);
        this->scheduler->evaluate(&req);
        this->evaluations++;
    }

    return new_node;
}

void PuctEvaluator::checkDrawStates(const PuctNode* node, PuctNode* next) {
    bool repeated = false;
    for (int ii=0; ii<this->number_repeat_states_draw; ii++) {
        if (node == nullptr) {
            break;
        }

        if ((*this->masked_bs_equals)(node->getBaseState(),
                                      next->getBaseState())) {
            repeated = true;
            break;
        }

        node = node->parent;
    }

    if (repeated) {
        for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
            next->setCurrentScore(ii, this->repeat_states_score);
        }

        next->is_finalised = true;
        //K273::l_verbose("setting node to finalised for repeated state @ depth %d", next->game_depth);
    }
}

float PuctEvaluator::getPuctConstant(PuctNode* node, int depth) const {
    // XXX remove this method
    return this->conf->puct_constant;
}

void PuctEvaluator::backPropagate(float* new_scores) {
    const int start_index = this->path.size() - 1;

    // back propagation:
    for (int index=start_index; index >= 0; index--) {
        const PathElement& cur = this->path[index];

        const float visits = cur.to_node->visits;

        for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
            const float score = ((visits * cur.to_node->getCurrentScore(ii) + new_scores[ii]) /
                                 (visits + 1.0));

            cur.to_node->setCurrentScore(ii, score);
        }

        cur.to_node->visits++;
    }
}

PuctNodeChild* PuctEvaluator::selectChild(PuctNode* node, int depth) {
    ASSERT(!node->isTerminal());

    if (node->num_children == 1) {
        return node->getNodeChild(this->sm->getRoleCount(), 0);
    }

    if (depth < 2) {
        this->setDirichletNoise(node);
    }

    float puct_constant = this->conf->puct_constant;
    float sqrt_node_visits = std::sqrt(node->visits + 1);
    float prior_score = this->priorScore(node, depth);

    // get best:
    float best_score = -1;
    PuctNodeChild* best_child = nullptr;

    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);

        float child_visits = 0.0f;

        float node_score = prior_score;

        float child_pct = c->policy_prob;
        if (c->to_node != nullptr) {
            PuctNode* cn = c->to_node;
            child_visits = cn->visits;
            node_score = cn->getCurrentScore(node->lead_role_index);

            // ensure finalised are enforced more than other nodes (network can return 1.0f for
            // basically dumb moves, if it thinks it will win regardless)
            // XXX isn't this redundant now, cause of 1.05 for wins?
            if (cn->is_finalised) {
                if (node_score > 0.99) {
                    if (depth > 0) {
                        return c;
                    }

                    node_score *= 1.0f + puct_constant;

                } else {
                    // no more exploration for you
                    child_pct = 0.0;
                }
            }
        }

        float cv = child_visits + 1;
        float puct_score = puct_constant * child_pct * (sqrt_node_visits / cv);

        // end product
        float score = node_score + puct_score;

        // use for debug/display
        c->debug_node_score = node_score;
        c->debug_puct_score = puct_score;

        if (score > best_score) {
            best_child = c;
            best_score = score;
        }
    }

    ASSERT(best_child != nullptr);
    return best_child;
}

int PuctEvaluator::treePlayout() {
    PuctNode* current = this->root;
    ASSERT(current != nullptr && !current->isTerminal());

    int tree_playout_depth = 0;

    this->path.clear();
    float scores[this->sm->getRoleCount()];

    PuctNodeChild* child = nullptr;

    while (true) {
        ASSERT(current != nullptr);
        this->path.emplace_back(child, current);

        // End of the road
        if (current->isTerminal()) {
            break;
        }

        // Choose selection
        PuctNodeChild* child = this->selectChild(current, tree_playout_depth);

        // if does not exist, then create it (will incur a nn prediction)
        if (child->to_node == nullptr) {
            this->expandChild(current, child, true);

            if (this->number_repeat_states_draw > 0) {
                this->checkDrawStates(current, child->to_node);
            }

            current = child->to_node;

            // special case if number of children is 1, we just bypass it and inherit next value
            if (!current->is_finalised && current->num_children == 1) {
                tree_playout_depth++;
                continue;
            }

            this->path.emplace_back(child, current);
            break;
        }

        current = child->to_node;
        tree_playout_depth++;
    }

    for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
        scores[ii] = current->getCurrentScore(ii);
    }

    this->backPropagate(scores);
    return tree_playout_depth;
}

void PuctEvaluator::playoutLoop(int max_evaluations, double end_time) {
    int max_depth = -1;
    int total_depth = 0;

    // configurable XXX?  Will only run at the very end of the game, and it is really only here so
    // we exit in a small finite amount of time

    // XXX normally constrained by evaluations anyways
    int max_iterations = max_evaluations * 100;

    if (max_evaluations < 0) {
        max_iterations = INT_MAX;
    }

    int iterations = 0;
    this->evaluations = 0;
    double start_time = K273::get_time();

    double next_report_time = -1;

    if (this->conf->matchmode) {
        next_report_time = K273::get_time() + 2.5;
    }

    while (iterations < max_iterations) {
        if (max_evaluations > 0 && this->evaluations > max_evaluations) {

            if (this->converged(this->root)) {
                break;

            } else {
                const int max_convergence_evaluations = max_evaluations * this->conf->evaluation_multipler_to_convergence;
                if (this->evaluations > max_convergence_evaluations) {
                    break;
                }
            }
        }

        if (end_time > 0 && K273::get_time() > end_time) {
            break;
        }

        int depth = this->treePlayout();
        max_depth = std::max(depth, max_depth);
        total_depth += depth;

        iterations++;

        if (next_report_time > 0 && K273::get_time() > next_report_time) {
            next_report_time = K273::get_time() + 2.5;

            const PuctNodeChild* best = this->chooseTopVisits(this->root);
            if (best->to_node != nullptr) {
                const int our_role_index = this->root->lead_role_index;
                const int choice = best->move.get(our_role_index);
                K273::l_info("Evals %d/%d, depth %.2f/%d, best: %.4f, move: %s",
                             this->evaluations, iterations,
                             total_depth / float(iterations), max_depth,
                             best->to_node->getCurrentScore(our_role_index),
                             this->sm->legalToMove(our_role_index, choice));
            }
        }
    }

    if (this->conf->verbose) {
        if (iterations) {
            K273::l_info("Time taken for %d/%d evaluations/iterations in %.3f seconds",
                         this->evaluations, iterations, K273::get_time() - start_time);

            K273::l_debug("The average depth explored: %.2f, max depth: %d",
                          total_depth / float(iterations), max_depth);
        } else {
            K273::l_debug("Did no iterations.");
        }
    }
}

PuctNode* PuctEvaluator::fastApplyMove(const PuctNodeChild* next) {
    ASSERT(this->initial_root != nullptr);
    ASSERT(this->root != nullptr);

    this->all_chained_nodes.push_back(this->root);

    PuctNode* new_root = nullptr;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);

        if (c == next) {
            ASSERT(new_root == nullptr);
            if (c->to_node == nullptr) {
                this->expandChild(this->root, c);
            }

            this->moves.push_back(c);

            new_root = c->to_node;

        } else {
            if (c->to_node != nullptr) {
                this->removeNode(c->to_node);

                // avoid a double delete at end of game
                c->to_node = nullptr;
            }
        }
    }

    ASSERT(new_root != nullptr);

    this->root = new_root;
    this->game_depth++;

    return this->root;
}

void PuctEvaluator::applyMove(const GGPLib::JointMove* move) {
    // XXX this is only here for the player.  We should probably have a player class, and not
    // simplify code greatly.
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);
        if (c->move.equals(move)) {
            this->fastApplyMove(c);
            break;
        }
    }

    ASSERT(this->root != nullptr);

    if (this->conf->verbose) {
        if (this->root->isTerminal()) {
            for (auto child : this->moves) {
                K273::l_info("Move made %s",
                             PuctNode::moveString(child->move, this->sm).c_str());
            }
        }
    }
}

void PuctEvaluator::reset(int game_depth) {
    // really free all
    if (this->initial_root != nullptr) {
        this->removeNode(this->initial_root);
        this->initial_root = nullptr;
        this->root = nullptr;
    }

    if (this->number_of_nodes) {
        K273::l_warning("Number of nodes not zero %d", this->number_of_nodes);
    }

    if (this->node_allocated_memory) {
        K273::l_warning("Leaked memory %ld", this->node_allocated_memory);
    }

    // these dont own the memory, so can just clear
    this->moves.clear();
    this->all_chained_nodes.clear();

    // this is the only place we set game_depth
    this->game_depth = game_depth;
}

PuctNode* PuctEvaluator::establishRoot(const GGPLib::BaseState* current_state) {
    ASSERT(this->root == nullptr && this->initial_root == nullptr);

    if (current_state == nullptr) {
        current_state = this->sm->getInitialState();
    }

    this->initial_root = this->root = this->createNode(nullptr, current_state);
    this->initial_root->game_depth = this->game_depth;

    ASSERT(!this->root->isTerminal());
    return this->root;
}

const PuctNodeChild* PuctEvaluator::onNextMove(int max_evaluations, double end_time) {
    ASSERT(this->root != nullptr && this->initial_root != nullptr);

    if (this->conf->root_expansions_preset_visits > 0) {
        int number_of_expansions = 0;
        for (int ii=0; ii<this->root->num_children; ii++) {
            PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);

            if (c->to_node == nullptr) {
                this->expandChild(this->root, c);

                // should be traversal on child XXX wait for puctplus
                c->to_node->visits = std::max(c->to_node->visits,
                                              this->conf->root_expansions_preset_visits);

                number_of_expansions++;
            }
        }
    }

    this->playoutLoop(max_evaluations, end_time);

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
    if (node == nullptr) {
        node = this->root;
    }

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

bool PuctEvaluator::converged(const PuctNode* node) const {
    if (node == nullptr) {
        return true;
    }

    auto children = PuctNode::sortedChildren(node, this->sm->getRoleCount());

    if (children.size() >= 2) {
        PuctNode* n0 = children[0]->to_node;
        PuctNode* n1 = children[1]->to_node;
        if (n0 != nullptr && n1 != nullptr) {
            const int role_index = node->lead_role_index;

            if (n0->getCurrentScore(role_index) > n1->getCurrentScore(role_index)) {
                // hardcode, but it is ok,  Just needs to actually move a little beyond 0.
                if (n0->visits > n1->visits + 8) {
                    return true;
                }
            }

            return false;
         }

    }

    return true;
}

const PuctNodeChild* PuctEvaluator::chooseTopVisits(const PuctNode* node) const {
    if (node == nullptr) {
        return nullptr;
    }

    auto children = PuctNode::sortedChildren(node, this->sm->getRoleCount());


    // compare top two.  This is a heuristic to cheaply check if the node hasn't yet converged and
    // chooses the one with the best score.  It isn't very accurate, the only way to get 100%
    // accuracy is to keep running for longer, until it cleanly converges.  This is a best guess for now.
    if (this->conf->top_visits_best_guess_converge_ratio > 0 && children.size() >= 2) {
        PuctNode* n0 = children[0]->to_node;
        PuctNode* n1 = children[1]->to_node;

        if (n0 != nullptr && n1 != nullptr) {
            const int role_index = node->lead_role_index;

            if (n1->visits > n0->visits * this->conf->top_visits_best_guess_converge_ratio &&
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
    float temperature = this->getTemperature();
    if (temperature < 0) {
        return this->chooseTopVisits(node);
    }

    // subtle: when the visits is low (like 0), we want to use the policy part of the
    // distribution. By using linger here, we get that behaviour.
    Children dist;
    if (!this->conf->dirichlet_noise_alpha && root->visits < root->num_children) {
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
    float node_visits = node->visits + 0.001 * node->num_children;

    // add some smoothness.  This also works for the case when doing no evaluations (ie
    // onNextMove(0)), as the node_visits == 0 and be uniform.
    float linger_pct = 0.01f;

    float total_probability = 0.0f;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        float child_visits = child->to_node ? child->to_node->visits + 0.001f : 0.001f;
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

PuctNode* PuctEvaluator::jumpRoot(int depth) {
    ASSERT(depth >=0 && depth < (int) this->all_chained_nodes.size());
    this->root = this->all_chained_nodes[depth];
    return this->root;
}
