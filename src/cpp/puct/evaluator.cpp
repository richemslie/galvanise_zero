
#include "puct/evaluator.h"
#include "puct/node.h"

#include "scheduler.h"

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

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::PuctEvaluator(GGPLib::StateMachineInterface* sm,
                             const PuctConfig* conf, NetworkScheduler* scheduler) :
    sm(sm),
    basestate_expand_node(nullptr),
    conf(conf),
    scheduler(scheduler),
    identifier("PuctEvaluator"),
    game_depth(0),
    evaluations(0),
    initial_root(nullptr),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0) {

    this->basestate_expand_node = this->sm->newBaseState();
}

PuctEvaluator::~PuctEvaluator() {
    free(this->basestate_expand_node);

    this->reset();
    delete this->conf;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::updateConf(const PuctConfig* conf) {
    this->conf = conf;
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

void PuctEvaluator::lookAheadTerminals(PuctNode* node) {
    ASSERT(!node->checked_for_finals);
    node->checked_for_finals = true;

    if (node->is_finalised) {
        return;
    }

    const int role_count = this->sm->getRoleCount();

    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(role_count, ii);

        this->sm->updateBases(node->getBaseState());

        if (c->to_node != nullptr) {
            return;
        }

        // apply move
        this->sm->nextState(&c->move, this->basestate_expand_node);
        this->sm->updateBases(this->basestate_expand_node);

        if (this->sm->isTerminal()) {
            PuctNode* terminal_node = this->createNode(node, this->basestate_expand_node);
            terminal_node->game_depth = node->game_depth + 1;

            ASSERT(terminal_node->is_finalised);

            c->to_node = terminal_node;

            // XXX what to do if lead_role_index is indeterminate?
            float score = terminal_node->getFinalScore(node->lead_role_index);
            if (score > 0.99) {
                for (int jj=0; jj<role_count; jj++) {
                    node->setFinalScore(jj, terminal_node->getFinalScore(jj));
                    node->setCurrentScore(jj, terminal_node->getCurrentScore(jj));
                }

                node->is_finalised = true;
                return;
            }
        }
    }
}

void PuctEvaluator::expandChild(PuctNode* parent, PuctNodeChild* child) {
    // update the statemachine
    this->sm->updateBases(parent->getBaseState());
    this->sm->nextState(&child->move, this->basestate_expand_node);

    // create node
    PuctNode* new_node = this->createNode(parent, this->basestate_expand_node);
    child->to_node = new_node;
    parent->num_children_expanded++;
    new_node->game_depth = parent->game_depth + 1;
}

PuctNode* PuctEvaluator::createNode(PuctNode* parent, const GGPLib::BaseState* state) {
    PuctNode* new_node = PuctNode::create(parent, state, this->sm);
    this->addNode(new_node);

    if (!new_node->is_finalised) {
        // goodbye kansas
        this->scheduler->evaluateNode(this, new_node);
        this->evaluations++;
    }

    return new_node;
}

bool PuctEvaluator::setDirichletNoise(int depth) {
    // set dirichlet noise on root?


    if (depth != 0) {
        return false;
    }

    if (this->conf->dirichlet_noise_alpha < 0) {
        return false;
    }

    // dont always run dirichlet_noise (XXX keep this?  add option?  drop for now)
    //if (this->rng.get() > 0.5) {
    //    return false;
    // }

    std::gamma_distribution<float> gamma(this->conf->dirichlet_noise_alpha, 1.0f);

    float total_noise = 0.0f;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);
        c->dirichlet_noise = gamma(this->rng);

        total_noise += c->dirichlet_noise;
    }

    // fail if we didn't produce any noise
    if (total_noise < std::numeric_limits<float>::min()) {
        return false;
    }

    // normalize:
    for (int ii=0; ii<this->root->num_children; ii++) {
       PuctNodeChild* c = this->root->getNodeChild(this->sm->getRoleCount(), ii);
       c->dirichlet_noise /= total_noise;
    }

    return true;
}

float PuctEvaluator::getPuctConstant(PuctNode* node) const {
    float constant = this->conf->puct_constant_after;
    int num_expansions = node == this->root ? this->conf->puct_before_root_expansions : this->conf->puct_before_expansions;

    if (node->num_children_expanded == node->num_children ||
        node->num_children_expanded < num_expansions) {
        constant = this->conf->puct_constant_before;
    }

    return constant;
}

void PuctEvaluator::backPropagate(float* new_scores) {
    const int role_count = this->sm->getRoleCount();
    const int start_index = this->path.size() - 1;

    bool only_once = true;

    // back propagation:
    const PathElement* prev = nullptr;

    for (int index=start_index; index >= 0; index--) {
        const PathElement& cur = this->path[index];

        if (only_once && !cur.to_node->is_finalised && cur.to_node->lead_role_index >= 0) {
            only_once = false;

            const PuctNodeChild* best = nullptr;
            {
                float best_score = -1;
                bool more_to_explore = false;
                for (int ii=0; ii<cur.to_node->num_children; ii++) {
                    const PuctNodeChild* c = cur.to_node->getNodeChild(role_count, ii);

                    if (c->to_node != nullptr && c->to_node->is_finalised) {
                        float score = c->to_node->getCurrentScore(cur.to_node->lead_role_index);
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
                    cur.to_node->setCurrentScore(ii, best->to_node->getCurrentScore(ii));
                }

                cur.to_node->is_finalised = true;
            }
        }

        if (prev != nullptr) {
        }


        if (cur.to_node->is_finalised) {
            for (int ii=0; ii<role_count; ii++) {
                new_scores[ii] = cur.to_node->getCurrentScore(ii);
            }

        } else {
            const int some_min_max_visit_config = 8;

            if (some_min_max_visit_config > 0 &&
                cur.to_node->visits > some_min_max_visit_config
                && prev != nullptr) {

                const PuctNodeChild* best = nullptr;
                {
                    // find the best
                    int best_visits = -1;

                    for (int ii=0; ii<cur.to_node->num_children; ii++) {
                        const PuctNodeChild* c = cur.to_node->getNodeChild(role_count, ii);
                        if (c->to_node != nullptr && c->to_node->visits > best_visits) {
                            best = c;
                            best_visits = c->to_node->visits;
                        }
                    }
                }

                ASSERT(best != nullptr && best->to_node != nullptr);

                if (prev->child != best && prev->to_node->visits != best->to_node->visits) {

                    const int role_index = cur.to_node->lead_role_index;

                    if (role_index != -1) {
                        float best_score = best->to_node->getCurrentScore(role_index);
                        bool improving = new_scores[role_index] > best_score;

                        // if it looks like it improving, then let it go
                        // i mean, I am all for exploration - but does it have to mess up the tree
                        // in the process? ...
                        if (!improving) {
                            for (int ii=0; ii<this->sm->getRoleCount(); ii++) {

                                new_scores[ii] = (0.6 * best->to_node->getCurrentScore(ii) +
                                                  0.4 * new_scores[ii]);
                            }
                        }
                    }
                }
            }

            for (int ii=0; ii<this->sm->getRoleCount(); ii++) {
                float score = (cur.to_node->visits * cur.to_node->getCurrentScore(ii) + new_scores[ii]) / (cur.to_node->visits + 1.0);
                cur.to_node->setCurrentScore(ii, score);
            }
        }

        cur.to_node->visits++;
        prev = &cur;
    }
}

PuctNodeChild* PuctEvaluator::selectChild(PuctNode* node, int depth) {
    ASSERT(!node->isTerminal());

    const double game_expected_depth = 60;
    if (!node->checked_for_finals) {
        // chance can easily become greater than one here
        double chance = (node->game_depth / game_expected_depth + node->visits / 40.0);

        if (this->rng.get() > (1 - chance)) {
            this->lookAheadTerminals(node);
        }
    }

    bool do_dirichlet_noise = this->setDirichletNoise(depth);

    float puct_constant = this->getPuctConstant(node);
    float sqrt_node_visits = ::sqrt(node->visits + 1);

    // get best
    PuctNodeChild* best_child = nullptr;
    float best_score = -1;

    /*
      // It is a good idea to keep this code, knowing what our noise looks like for different games is
      // an important configuration step
    if (this->conf->verbose && do_dirichlet_noise) {
        std::string debug_dirichlet_noise = "dirichlet_noise = ";
        for (int ii=0; ii<node->num_children; ii++) {
            PuctNodeChild* c = node->getNodeChild(this->role_count, ii);
            debug_dirichlet_noise += K273::fmtString("%.3f", c->dirichlet_noise);
            if (ii != node->num_children - 1) {
                debug_dirichlet_noise += ", ";
            }
        }

        K273::l_info(debug_dirichlet_noise);
    }
    */

    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->sm->getRoleCount(), ii);

        float child_visits = 0.0f;

        // prior... (alpha go zero said 0 but there score ranges from [-1,1]
        float node_score = 0.0f;

        if (c->to_node != nullptr) {
            PuctNode* cn = c->to_node;
            child_visits = (float) cn->visits;
            node_score = cn->getCurrentScore(node->lead_role_index);

            // ensure terminals are enforced more than other nodes (network can return 1.0f for
            // basically dumb moves, if it thinks it will win regardless)
            if (cn->is_finalised) {

                // return straight away at lower levels, for depth zero we still want a
                // probability distribution
                if (depth > 0 && node_score > 0.99) {
                    c->debug_node_score = node_score;
                    c->debug_puct_score = -1;
                    return c;
                }

                node_score *= 1.02f;
            }
        }

        float child_pct = c->policy_prob;

        if (do_dirichlet_noise) {
            float noise_pct = this->conf->dirichlet_noise_pct;
            child_pct = (1.0f - noise_pct) * child_pct + noise_pct * c->dirichlet_noise;
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
            this->expandChild(current, child);
            current = child->to_node;
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
    int max_iterations = max_evaluations * 42;

    if (max_evaluations < 0) {
        max_iterations = max_evaluations = INT_MAX;
    }

    int iterations = 0;
    this->evaluations = 0;
    double start_time = K273::get_time();

    while (iterations < max_iterations) {

        if (this->evaluations > max_evaluations) {
            break;
        }

        if (end_time > 0 && K273::get_time() > end_time) {
            break;
        }

        int depth = this->treePlayout();
        max_depth = std::max(depth, max_depth);
        total_depth += depth;

        iterations++;
    }

    if (this->conf->verbose) {
        if (iterations) {
            K273::l_info("Time taken for %d/%d evaluations/iterations %.3f",
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

void PuctEvaluator::reset() {
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

    this->game_depth = 0;

    // these dont own the memory, so can just clear
    this->moves.clear();
    this->all_chained_nodes.clear();
}

PuctNode* PuctEvaluator::establishRoot(const GGPLib::BaseState* current_state, int game_depth) {
    // needed for temperature
    this->game_depth = game_depth;

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

    if (max_evaluations > 0 && this->conf->verbose) {
        this->logDebug(choice);
    }

    return choice;
}

float PuctEvaluator::getTemperature() const {
    if (this->game_depth > this->conf->depth_temperature_stop) {
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

    // check top two comparison
    if (children.size() >= 2) {
        PuctNode* n0 = children[0]->to_node;
        PuctNode* n1 = children[1]->to_node;

        if (n0 != nullptr && n1 != nullptr) {
            const int role_index = node->lead_role_index;

            if (n1->getCurrentScore(role_index) > n0->getCurrentScore(role_index)) {
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

    // add some smoothness

    float linger_pct = 0.1f;

    float total_probability = 0.0f;
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->sm->getRoleCount(), ii);
        int child_visits = child->to_node ? child->to_node->visits + 0.1f : 0.1f;
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
