#include "puct/evaluator.h"
#include "puct/node.h"

#include "supervisorbase.h"

#include <k273/util.h>
#include <k273/logging.h>
#include <k273/exception.h>

#include <cmath>
#include <climits>
#include <unistd.h>
#include <random>
#include <numeric>

using namespace GGPZero;

///////////////////////////////////////////////////////////////////////////////

PuctEvaluator::PuctEvaluator(PuctConfig* config, SupervisorBase* supervisor) :
    config(config),
    supervisor(supervisor),
    role_count(supervisor->getRoleCount()),
    identifier("PuctEvaluator"),
    game_depth(0),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0) {
}

PuctEvaluator::~PuctEvaluator() {

    if (this->root != nullptr) {
        this->removeNode(this->root);
        this->root = nullptr;
    }

    if (this->number_of_nodes) {
        K273::l_warning("Number of nodes not zero %d", this->number_of_nodes);
    }

    if (this->node_allocated_memory) {
        K273::l_warning("Leaked memory %ld", this->node_allocated_memory);
    }

    delete this->config;
}

///////////////////////////////////////////////////////////////////////////////

void PuctEvaluator::addNode(PuctNode* new_node) {
    this->number_of_nodes++;
    this->node_allocated_memory += new_node->allocated_size;
}

void PuctEvaluator::removeNode(PuctNode* node) {
    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* child = node->getNodeChild(this->role_count, ii);
        if (child->to_node != nullptr) {
            this->removeNode(child->to_node);
        }

        child->to_node = nullptr;
    }

    this->node_allocated_memory -= node->allocated_size;

    free(node);
    this->number_of_nodes--;
}


void PuctEvaluator::expandChild(PuctNode* parent, PuctNodeChild* child) {
    PuctNode* new_node = this->supervisor->expandChild(this, parent, child);
    this->addNode(new_node);
    child->to_node = new_node;
    parent->num_children_expanded++;
}

PuctNode* PuctEvaluator::createNode(const GGPLib::BaseState* state) {
    PuctNode* new_node = this->supervisor->createNode(this, state);
    this->addNode(new_node);
    return new_node;
}

bool PuctEvaluator::setDirichletNoise(int depth) {
    // set dirichlet noise on root?

    if (depth != 0) {
        return false;
    }

    if (this->config->dirichlet_noise_alpha < 0) {
        return false;
    }

    std::gamma_distribution<float> gamma(this->config->dirichlet_noise_alpha, 1.0f);

    float total_noise = 0.0f;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->role_count, ii);
        c->dirichlet_noise = gamma(this->random);
        total_noise += c->dirichlet_noise;
    }

    // fail if we didn't produce any noise
    if (total_noise < std::numeric_limits<float>::min()) {
        return false;
    }

    // normalize:
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->role_count, ii);
        c->dirichlet_noise /= total_noise;
    }

    return true;
}

double PuctEvaluator::getPuctConstant(PuctNode* node) const {
    double constant = this->config->puct_constant_after;
    int num_expansions = node == this->root ? this->config->puct_before_root_expansions : this->config->puct_before_expansions;

    if (node->num_children_expanded < num_expansions) {
        constant = this->config->puct_constant_before;
    }

    return constant;
}

void PuctEvaluator::backPropagate(double* new_scores) {
    const int start_index = this->path.size() - 1;

    // back propagation:
    for (int index=start_index; index >= 0; index--) {
        PuctNode* node = this->path[index];

        for (int ii=0; ii<this->role_count; ii++) {
            double score = (node->visits * node->getCurrentScore(ii) + new_scores[ii]) / (node->visits + 1.0);
            node->setCurrentScore(ii, score);
        }

        node->visits++;
    }
}

PuctNodeChild* PuctEvaluator::selectChild(PuctNode* node, int depth) {
    bool do_dirichlet_noise = this->setDirichletNoise(depth);

    double puct_constant = this->getPuctConstant(node);
    double sqrt_node_visits = ::sqrt(node->visits + 1);

    // get best
    PuctNodeChild* best_child = nullptr;
    double best_score = -1;

    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->role_count, ii);

        double child_visits = 0.0;

        // prior... (alpha go zero said 0 but there score ranges from [-1,1]
        double node_score = 0.0;

        if (c->to_node != nullptr) {
            PuctNode* cn = c->to_node;
            child_visits = (double) cn->visits;
            node_score = cn->getCurrentScore(node->lead_role_index);

            // ensure terminals are enforced more than other nodes (network can return 1.0 for
            // basically dumb moves, if it thinks it will win regardless)
            if (cn->isTerminal()) {
                node_score *= 1.02;
            }
        }

        float child_pct = c->policy_prob;

        if (do_dirichlet_noise) {
            double noise_pct = this->config->dirichlet_noise_pct;
            child_pct = (1.0 - noise_pct) * child_pct + noise_pct * c->dirichlet_noise;
        }

        double cv = child_visits + 1;
        double puct_score = puct_constant * child_pct * (sqrt_node_visits / cv);

        // end product
        double score = node_score + puct_score;

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
    double scores[this->role_count];

    while (true) {
        ASSERT(current != nullptr);
        this->path.push_back(current);

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
            this->path.push_back(current);
            break;
        }

        current = child->to_node;
        tree_playout_depth++;
    }

    for (int ii=0; ii<this->role_count; ii++) {
        scores[ii] = current->getCurrentScore(ii);
    }

    this->backPropagate(scores);
    return tree_playout_depth;
}


void PuctEvaluator::playoutLoop(int max_iterations) {
    int max_depth = -1;
    int total_depth = 0;
    int iterations = 0;

    double start_time = K273::get_time();
    if (max_iterations < 0) {
        // XXX sys.maxint
        max_iterations = INT_MAX;
    }

    while (iterations < max_iterations) {
        int depth = this->treePlayout();
        max_depth = std::max(depth, max_depth);
        total_depth += depth;

        iterations++;
    }

    if (this->config->verbose) {
        if (iterations) {
            K273::l_info("Time taken for %d iteratons %.3f", iterations, K273::get_time() - start_time);

            K273::l_debug("The average depth explored: %.2f, max depth: %d",
                          total_depth / double(iterations), max_depth);
        } else {
            K273::l_debug("Did no iterations.");
        }
    }
}

PuctNode* PuctEvaluator::fastApplyMove(PuctNodeChild* next) {
    ASSERT(this->root != nullptr);

    PuctNode* new_root = nullptr;
    for (int ii=0; ii<this->root->num_children; ii++) {
        PuctNodeChild* c = this->root->getNodeChild(this->role_count, ii);

        if (c == next) {
            new_root = c->to_node;

            // removeNode() is recursive, we must disconnect it from the tree here before calling
            c->to_node = nullptr;
            break;
        }
    }

    this->removeNode(this->root);

    ASSERT(new_root);
    this->root = new_root;
    return this->root;
}

void PuctEvaluator::reset() {
    // free all
    if (this->root != nullptr) {
        this->removeNode(this->root);
        this->root = nullptr;
    }
}

PuctNode* PuctEvaluator::establishRoot(const GGPLib::BaseState* current_state, int game_depth) {
    // needed for temperature
    this->game_depth = game_depth;

    if (this->config->verbose) {
        K273::l_verbose("Debug @ depth %d", game_depth);
    }

    ASSERT (this->root == nullptr);

    this->root = this->createNode(current_state);
    ASSERT(!this->root->isTerminal());

    return this->root;
}

PuctNodeChild* PuctEvaluator::onNextNove(int max_iterations) {
    this->playoutLoop(max_iterations);

    PuctNodeChild* choice = this->chooseTopVisits(this->root);

    if (this->config->verbose) {
        this->logDebug();
    }

    return choice;
}

void PuctEvaluator::logDebug() {
    PuctNode* cur = this->root;
    for (int ii=0; ii<this->config->max_dump_depth; ii++) {
        std::string indent = "";
        for (int jj=ii-1; jj>=0; jj--) {
            if (jj > 0) {
                indent += "    ";
            } else {
                indent += ".   ";
            }
        }

        PuctNodeChild* next_choice;
        if (cur->num_children == 0) {
            next_choice = nullptr;
        } else {
            next_choice = this->chooseTopVisits(cur);
        }

        this->supervisor->dumpNode(cur, next_choice, indent);
        cur = next_choice->to_node;
        if (cur == nullptr) {
            break;
        }
    }
}

PuctNodeChild* PuctEvaluator::chooseTopVisits(PuctNode* node) {
    int best_visits = -1;
    PuctNodeChild* selection = nullptr;

    for (int ii=0; ii<node->num_children; ii++) {
        PuctNodeChild* c = node->getNodeChild(this->role_count, ii);

        if (c->to_node != nullptr && c->to_node->visits > best_visits) {
            best_visits = c->to_node->visits;
            selection = c;
        }
    }

    // failsafe - random
    if (selection == nullptr) {
        selection = node->getNodeChild(this->role_count, this->random.getWithMax(node->num_children));
    }

    return selection;
}
