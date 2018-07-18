#include "mcts.h"

#include "config.h"
#include "worker.h"

#include <k273/util.h>
#include <k273/logging.h>
#include <k273/exception.h>

#include <cmath>
#include <string>
#include <unistd.h>

using namespace K273;
using namespace GGPLib;
using namespace PlayerMcts;

static bool fequals(double x, double y) {
    return std::fabs(x - y) < 0.001;
}

static bool wins(double score) {
    return score > 0.999;
}

///////////////////////////////////////////////////////////////////////////////

Player::Player(StateMachineInterface* sm, int player_role_index, const Config* config) :
    PlayerBase(sm, player_role_index),
    config(config),
    root(nullptr),
    number_of_nodes(0),
    node_allocated_memory(0) {
    this->playout_stats.reset();
}

Player::~Player() {
    if (this->root != nullptr) {
        this->releaseNodes(this->root);
        for (Node* n : this->garbage) {
            this->removeNode(n);
        }

        if (this->root->ref_count == 1) {
            this->removeNode(this->root);
        }
    }

    this->garbage.clear();
    this->lookup.clear();

    for (Worker* worker : this->workers) {
        worker->getThread()->kill();
        delete worker->getThread();
        delete worker;
    }

    this->workers.clear();

    if (this->number_of_nodes) {
        K273::l_warning("Number of nodes not zero %d", this->number_of_nodes);
    }

    if (this->node_allocated_memory) {
        K273::l_warning("Leaked memory %ld", this->node_allocated_memory);
    }
}

///////////////////////////////////////////////////////////////////////////////

Node* Player::lookupNode(const BaseState* bs) {
    auto found = this->lookup.find(bs);
    if (found != this->lookup.end()) {
        Node* result = found->second;
        result->ref_count++;
        return result;
    }

    return nullptr;
}

Node* Player::createNode(const BaseState* bs) {
    const int role_count = this->sm->getRoleCount();
    Node* new_node = Node::create(role_count,
                                  this->our_role_index,
                                  this->config->initial_ucb_constant,
                                  bs,
                                  this->sm);

    // XXX if this triggers then need to normalise the score.  This is very, very unlikely to happen
    // - only called for root nodes.
    ASSERT (!new_node->is_finalised);

    this->lookup[new_node->getBaseState()] = new_node;
    this->number_of_nodes++;
    this->node_allocated_memory += new_node->allocated_size;

    return new_node;
}

void Player::removeNode(Node* n) {
    const BaseState* bs = n->getBaseState();
    this->lookup.erase(bs);
    this->node_allocated_memory -= n->allocated_size;

    free(n);
    this->number_of_nodes--;
}

void Player::releaseNodes(Node* current) {
    int role_count = this->sm->getRoleCount();
    for (int ii=0; ii<current->num_children; ii++) {
        NodeChild* child = current->getNodeChild(role_count, ii);
        if (child->to_node != nullptr) {

            Node* next_node = child->to_node;

            // wah a cycle...
            if (next_node->ref_count == 0) {
                K273::l_warning("A cycle was found in Player::releaseNodes() skipping");
                continue;
            }

            child->to_node = nullptr;

            ASSERT (next_node->ref_count > 0);
            next_node->ref_count--;
            if (next_node->ref_count == 0) {
                this->releaseNodes(next_node);
                this->garbage.push_back(next_node);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

bool Player::selectChild(Node* node, Path::Selected& path, int depth) {
    ASSERT (!node->is_finalised);

    const int role_count = this->sm->getRoleCount();
    int lead_role_index = node->lead_role_index;
    if (lead_role_index < 0) {
        lead_role_index = this->our_role_index;
    }

    double best_explore_score = -1000000;
    double best_final_score = -1000000;
    double best_exploit_score = -1000000;

    NodeChild* best_child = nullptr;
    NodeChild* best_final_child = nullptr;
    NodeChild* best_child_exploitation = nullptr;

    const double exploration_constant = node->select_ucb_constant;

    const int random_counts = std::max(this->config->select_random_move_count, node->num_children / 4);

    for (int ii=0; ii<node->num_children; ii++) {
        NodeChild* c = node->getNodeChild(role_count, ii);

        // skip unselectables
        if (c->unselectable) {
            continue;

        } else if (c->to_node != nullptr &&
                   (c->to_node->num_children > 0 &&
                    c->to_node->unselectable_count == c->to_node->num_children)) {
            continue;
        }

        const int inflight_visits = c->to_node != nullptr ? c->to_node->inflight_visits : 0;
        double exploration_bonus = exploration_constant;

        double search_score;
        if (node->visits < random_counts) {
            const double rand_size = 100 * (this->rng.getWithMax(1000) + 1);
            search_score = 1.0 + 1.0 / rand_size;

        } else {
            const double traversals = std::max(1.0, (double) c->traversals);

            if (c->to_node != nullptr) {

                if (c->to_node->is_finalised) {
                    search_score = c->to_node->getScore(lead_role_index);

                    if (search_score > best_final_score) {
                        best_final_score = search_score;
                        best_final_child = c;
                    }

                    // special case - early bail, since there is no point exploring other branches.
                    if (wins(search_score)) {
                        // choose best_final_child at end
                        best_child = nullptr;
                        break;
                    }

                    // try not to select finalised which doesn't provide any info.
                    if (fequals(search_score, 0) || role_count == 1) {
                        continue;
                    }

                    // no exploration bonus for you
                    exploration_bonus = -0.1;

                } else {
                    search_score = c->to_node->getScore(lead_role_index);

                    if (inflight_visits) {
                        // exploration bonus takes into account inflight visits
                        exploration_bonus = (exploration_constant * node->sqrt_log_visits) / sqrt(traversals + inflight_visits);

                        // simulate worst possible result for inflight_visits, encourages more
                        // exploration with inflight visits
                        search_score = (search_score * traversals) / (traversals + inflight_visits);

                    } else {
                        // fast path - precomputed values
                        exploration_bonus = exploration_constant * node->sqrt_log_visits * c->inv_sqrt_traversals;
                    }
                }

                if (search_score > best_exploit_score) {
                    best_exploit_score = search_score;
                    best_child_exploitation = c;
                }

            } else {
                // max score
                search_score = 1.0;

                // add some randomness
                const double rand_size = 100 * (this->rng.getWithMax(1000) + 1);
                search_score += 1.0 / rand_size;

                // simulate worst possible result for inflight_visits, encourages exploration
                if (inflight_visits > 0) {
                    search_score = (search_score * traversals) / (traversals + inflight_visits);
                }
            }
        }

        const double score = search_score + exploration_bonus;

        if (score > best_explore_score) {
            best_explore_score = score;
            best_child = c;
        }
    }

    if (best_child == nullptr) {
        best_child = best_final_child;

        // in this case all children are stuck waiting for expansions (ideally this shouldn't
        // happen often)
        if (best_child == nullptr) {
            return false;
        }
    }

    path.add(node, best_child, best_child == best_child_exploitation);
    return true;
}

bool Player::selectChildAdjust(Node* node, Path::Selected& path, int depth) {
    if (!this->selectChild(node, path, depth)) {
        // handled above
        return false;
    }

    auto last = path.getLast();
    ASSERT (last->node == node && last->selection != nullptr);

    if (last->exploitation && node->visits > 2 * node->num_children) {

        double percent_visits = ((double) last->selection->traversals) / ((double) node->visits);

        // Check if already above, since there is cases where select_ucb_constant is much greater
        // than node->select_ucb_constant

        int v = std::min(0.2, std::max(0, (node->num_children - 10)) * 0.01);
        if (percent_visits > 0.5 - v) {
            // basically similar to aggressively choosing a final winning move for us
            node->select_ucb_constant *= 1.0025;


            if (node->select_ucb_constant > this->config->upper_adjust_ucb_constant) {
                node->select_ucb_constant = this->config->upper_adjust_ucb_constant;
            }
        }

        if (percent_visits < 0.20) {
            // it may already be less for other reasons...  "dont go less" below will end up increasing it
            if (node->select_ucb_constant > this->config->lower_adjust_ucb_constant) {
                node->select_ucb_constant *= 0.9975;

                // don't go less
                if (node->select_ucb_constant < this->config->lower_adjust_ucb_constant) {
                    node->select_ucb_constant = this->config->lower_adjust_ucb_constant;
                }
            }
        }
    }

    return true;
}

//////////////////////////////////////////////////////////////////////

void Player::processAll() {
    /* done after main loop, not performance critical */

    while (true) {
        if (this->worker_event_queue.empty()) {
            if (this->workers_available.size() != this->config->thread_workers) {
                continue;
            }

            break;
        }

        this->processAny(false);
    }
}

Worker* Player::processAny(bool return_worker) {
    /* done after main loop, not performance critical */

    while (true) {
        Event* event = this->worker_event_queue.pop();
        if (event == nullptr) {
            return nullptr;
        }

        if (event->event_type == Event::EXPANSION) {
            this->expandTree(event->worker);

        }  else {
            ASSERT (event->event_type == Event::ROLLOUT);

            // this might not be a rollout at all, and indeed just a early finalised.
            if (event->worker->did_rollout) {
                this->playout_stats.rollout_accumulative_time += event->worker->time_for_rollout;
                this->playout_stats.rollouts++;
            }

            if (return_worker) {
                return event->worker;
            }

            // do backprop asap:
            this->backPropagate(event->worker->path, event->worker->score);

            // ready for next time
            event->worker->reset();
            this->workers_available.push(event->worker);
        }
    }
}

void Player::expandTree(Worker* worker) {
    const double expansion_start_time = get_rtc_relative_time();

    // get last selection from path
    Path::Element* last = worker->path.getLast();

    // this should all be blocked by unselectables.
    ASSERT (last->selection != nullptr);
    ASSERT (last->selection->to_node == nullptr);
    ASSERT (last->selection->traversals == 0);

    last->selection->unselectable = false;
    last->node->unselectable_count--;

    ASSERT (last->node->unselectable_count >= 0 && last->node->unselectable_count < last->node->num_children);

    // node to insert into tree
    Node* found = this->lookupNode(worker->new_node->getBaseState());

    if (found == nullptr) {
        found = worker->new_node;
        this->lookup[found->getBaseState()] = found;
        this->number_of_nodes++;
        this->node_allocated_memory += found->allocated_size;

    } else {
        // delete the new node
        free(worker->new_node);
        this->playout_stats.transpositions++;
    }

    // insert into tree
    last->selection->to_node = found;

    // add to path so can backPropagate correctly
    worker->path.add(found);

    // update stats
    this->playout_stats.creation_accumulative_time += worker->time_for_expansion;
    this->playout_stats.expansions_accumulative_time += get_rtc_relative_time() - expansion_start_time;
}

///////////////////////////////////////////////////////////////////////////////

void Player::backPropagate(Path::Selected& path, std::vector<double>& new_scores) {
    const double back_propagate_start_time = get_rtc_relative_time();

    const int role_count = this->sm->getRoleCount();

    const int start_index = path.size() - 1;

    bool only_once = true;

    // back propagation:
    for (int index=start_index; index >= 0; index--) {
        auto back = path.get(index);
        Node* back_node = back->node;

        // determine if can finalise this node?
        if (only_once && !back_node->is_finalised && back_node->lead_role_index >= 0) {
            only_once = false;

            const NodeChild* best = nullptr;
            {
                double best_score = -1;
                bool more_to_explore = false;
                for (int ii=0; ii<back_node->num_children; ii++) {
                    const NodeChild* c = back->node->getNodeChild(role_count, ii);

                    if (c->to_node != nullptr && c->to_node->is_finalised) {
                        double score = c->to_node->getScore(back_node->lead_role_index);
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
                if (wins(best_score)) {
                    if (more_to_explore) {
                        this->playout_stats.finalised_count_opportunist++;
                        more_to_explore = false;
                    }
                }

                if (more_to_explore) {
                    best = nullptr;
                }
            }

            if (best != nullptr) {
                for (int ii=0; ii<role_count; ii++) {
                    double score = best->to_node->getScore(ii);
                    back_node->setScore(ii, score);
                }

                // stats
                this->playout_stats.finalised_count++;
                if (back_node->lead_role_index == this->our_role_index) {
                    this->playout_stats.finalised_count_our_role++;
                }

                back_node->is_finalised = true;
            }
        }

        // ensure scores are correct - our rollout scores may be different from this, so update

        if (back_node->is_finalised) {
            for (int ii=0; ii<role_count; ii++) {
                new_scores[ii] = back_node->getScore(ii);
            }

        } else {
            for (int ii=0; ii<role_count; ii++) {
                double score = (back_node->visits * back_node->getScore(ii) + new_scores[ii]) / (back_node->visits + 1.0);
                back_node->setScore(ii, score);
            }
        }

        back_node->visits++;
        back_node->sqrt_log_visits = std::sqrt(std::log(back_node->visits + 1.0));

        if (back->selection != nullptr) {
            back->selection->traversals++;
            back->selection->inv_sqrt_traversals = std::sqrt(1.0 / (double) back->selection->traversals);
        }

        if (back_node->inflight_visits > 0) {
            back_node->inflight_visits--;
        }
    }

    this->playout_stats.back_propagate_accumulative_time += get_rtc_relative_time() - back_propagate_start_time;
}

int Player::treePlayout(Path::Selected& path, bool look_wide_at_start) {

    const double tree_playout_start_time = get_rtc_relative_time();

    int tree_playout_depth = 0;

    Node* current = this->root;

    while (true) {
        ASSERT (current != nullptr);

        current->inflight_visits++;

        // End of the road
        if (current->is_finalised) {
            path.add(current);
            break;
        }

        // Choose selection based on lead_role_index
        if (look_wide_at_start && tree_playout_depth == 0) {
            current->select_ucb_constant = std::max(this->config->lead_first_node_ucb_constant,
                                                    current->select_ucb_constant);

        }

        if (!this->selectChildAdjust(current, path, tree_playout_depth)) {
            // only get here when there was nothing to select.  This can happen when all the
            // threads are busy expanding nodes.
            return -1;
        }

        Node* next = path.getNextNode();

        if (next == nullptr) {
            auto last = path.getLast();
            last->selection->unselectable = true;
            current->unselectable_count++;
            tree_playout_depth++;

            // this is the breakpoint, need to expand the tree
            break;
        }

        current = next;

        tree_playout_depth++;
    }

    this->playout_stats.total_tree_playout_depth += tree_playout_depth;

    this->playout_stats.tree_playouts++;
    this->playout_stats.tree_playout_accumulative_time += get_rtc_relative_time() - tree_playout_start_time;

    return tree_playout_depth;
}

///////////////////////////////////////////////////////////////////////////////

NodeChild* Player::chooseBest(Node* node, bool warn) {
    if (node->num_children == 0) {
        return nullptr;
    }

    const int role_count = this->sm->getRoleCount();
    int lead_role_index = node->lead_role_index;
    if (lead_role_index < 0) {
        lead_role_index = this->our_role_index;
    }

    // finalised based on score.  Non-finalised based on visits.
    int best_visits = -1;
    double best_final_score = -1;
    double best_visits_final_score = -1;

    NodeChild* selection_score = nullptr;
    NodeChild* selection_visits = nullptr;

    double best_non_final_score = -1;

    // used for when best_non_final_score tie (choose one with most iterations)
    int best_non_final_score_visits = -1;

    NodeChild* selection_non_final_score = nullptr;
    for (int ii=0; ii<node->num_children; ii++) {
        NodeChild* c = node->getNodeChild(role_count, ii);

        // not sure there is anything better to do at this point
        if (c->to_node == nullptr) {
            continue;
        }

        if (c->to_node->is_finalised) {
            double final_score = c->to_node->getScore(lead_role_index);

            if (fequals(final_score, best_final_score)) {
                // grab with least iterations for final score
                ASSERT (best_visits_final_score >= 0);

                if (wins(final_score)) {
                    // shortest path
                    if (c->traversals < best_visits_final_score) {
                        best_visits_final_score = c->traversals;
                        selection_score = c;
                    }

                } else {
                    // longest path (maybe, just a heuristic to complicate things for opponent)
                    if (c->to_node->visits > best_visits_final_score) {
                        best_visits_final_score = c->to_node->visits;
                        selection_score = c;
                    }
                }

            } else if (final_score > best_final_score) {
                best_visits_final_score = c->to_node->visits;
                selection_score = c;
                best_final_score = final_score;
            }

            ASSERT (selection_score != nullptr);

        } else {
            double score = c->to_node->getScore(lead_role_index);
            if (score > best_non_final_score) {
                best_non_final_score = score;
                best_non_final_score_visits = c->traversals;
                selection_non_final_score = c;

            } else if (fequals(score, best_non_final_score)) {
                if (c->traversals > best_non_final_score_visits) {
                    best_non_final_score = score;
                    best_non_final_score_visits = c->traversals;
                    selection_non_final_score = c;
                }
            }

            if (c->traversals > best_visits) {
                best_visits = c->traversals;
                selection_visits = c;
            }
        }
    }

    if (wins(best_final_score)) {
        return selection_score;
    }

    NodeChild* selection = nullptr;
    if (selection_visits == nullptr) {
        selection = selection_score;

    } else if (selection_score == nullptr) {
        selection = selection_visits;
        if (this->config->selection_use_scores) {
            if (selection_non_final_score != selection) {
                if (warn) {
                    K273::l_warning("selection difference non_final versus visits - using scores");
                }

                selection = selection_non_final_score;
            }
        }

    } else {
        if (selection_score->to_node->getScore(lead_role_index) > selection_visits->to_node->getScore(lead_role_index)) {
            selection = selection_score;

        } else {
            selection = selection_visits;
            if (this->config->selection_use_scores) {
                if (selection_non_final_score != selection) {
                    if (warn) {
                        K273::l_warning("selection difference non_final versus visits - using scores");
                    }

                    selection = selection_non_final_score;
                }
            }
        }
    }

    // failsafe - random
    if (selection == nullptr) {
        selection = node->getNodeChild(role_count, this->rng.getWithMax(node->num_children));
    }

    return selection;
}

void Player::mainLoop(int playout_next_check_count, bool do_lead_first_node) {
    const double main_loop_start_time = get_rtc_relative_time();

    const int role_count = this->sm->getRoleCount();
    std::vector <double> new_scores;
    Worker* worker = nullptr;

    while (this->playout_stats.tree_playouts < playout_next_check_count) {

        if (worker == nullptr) {

            const double poll_start_time = get_rtc_relative_time();

            Event* event = this->worker_event_queue.pop();

            if (event != nullptr) {
                this->playout_stats.poll_available_workers_accumulative_time += get_rtc_relative_time() - poll_start_time;

                if (event->event_type == Event::EXPANSION) {
                    this->expandTree(event->worker);
                    continue;

                } else {
                    ASSERT (event->event_type == Event::ROLLOUT);

                    // this might not be a rollout at all, and indeed just a early finalised.
                    if (event->worker->did_rollout) {
                        this->playout_stats.rollout_accumulative_time += event->worker->time_for_rollout;
                        this->playout_stats.rollouts++;
                    }

                    worker = event->worker;

                    // do backprop asap:
                    this->backPropagate(event->worker->path, event->worker->score);
                    worker->reset();

                    // will reuse this worker
                }

            } else {
                if (!this->workers_available.empty()) {
                    worker = this->workers_available.front();
                    this->workers_available.pop();
                }

                this->playout_stats.poll_available_workers_accumulative_time += get_rtc_relative_time() - poll_start_time;

                if (worker == nullptr) {
                    this->playout_stats.main_loop_waiting++;
                    continue;
                }
            }
        }

        ASSERT (worker != nullptr && worker->is_reset);

        // actually do the tree playout.
        int tree_playout_depth = this->treePlayout(worker->path, do_lead_first_node);

        // failed to do a tree playout
        if (unlikely(tree_playout_depth == -1)) {
            worker->reset();
            this->workers_available.push(worker);
            worker = nullptr;
            this->playout_stats.total_unselectable_count++;
            continue;
        } else if (unlikely(tree_playout_depth == 0)) {
            break;
        }

        Path::Element* last = worker->path.getLast();
        if (unlikely(last->node->is_finalised)) {
            // ok, this has nothing to do with the worker thread itself.  The treePlayout() will
            // build up a path, to pass it on to the worker.  In this case the path ended with already finalised expanded node.
            // Hence the worker has nothing to do, using/abusing the PathSelected object on the worker.
            // In other words, this just indicates that a finalised node was hit during the
            // treePlayout().

            ASSERT (last->selection == nullptr);

            // Set the scores and call backPropagate()
            new_scores.clear();
            for (int ii=0; ii<role_count; ii++) {
                new_scores.push_back(last->node->getScore(ii));
            }

            this->backPropagate(worker->path, new_scores);

            // obviously don't need the worker thread to do anything - just reset and loop
            worker->reset();

        } else {
            const double prompt_start_time = get_rtc_relative_time();

            // Let the worker rip
            worker->getThread()->promptWorker();
            worker = nullptr;

            this->playout_stats.prompt_accumulative_time += get_rtc_relative_time() - prompt_start_time;
        }

        // worker passed to the top of the loop
    }

    if (worker != nullptr) {
        worker->reset();
        this->workers_available.push(worker);
    }

    this->playout_stats.main_loop_accumulative_time += get_rtc_relative_time() - main_loop_start_time;
}

///////////////////////////////////////////////////////////////////////////////

void Player::onMetaGaming(double end_time) {
    ASSERT (this->workers.size() == 0 && this->workers_available.size() == 0 &&
            this->worker_event_queue.pop() == nullptr);

    double enter_time = get_time();
    K273::l_info("entering onMetaGaming() with %.1f seconds", end_time - enter_time);

    for (unsigned int ii=0; ii<this->config->thread_workers; ii++) {
        Worker* worker = new Worker(&this->worker_event_queue,
                                    this->sm->dupe(),
                                    this->config, this->our_role_index);
        WorkerThread* worker_thread = new WorkerThread(worker);
        worker_thread->spawn();
        this->workers.push_back(worker);
    }

    K273::l_debug("created threads");

    // just builds tree for a bit (too long and will run out of memory)
    this->onNextMove(std::min(end_time - 5, get_time() + 30));
}

void Player::onApplyMove(JointMove* last_move) {

    this->game_depth++;
    K273::l_info("MCTS: game depth %d", this->game_depth);
    int number_of_nodes_before = this->number_of_nodes;

    // get a new root, and cleanup orphaned branches from tree
    int role_count = this->sm->getRoleCount();
    if (this->root != nullptr) {
        NodeChild* found_child = nullptr;

        // find the child in the root
        for (int ii=0; ii<this->root->num_children; ii++) {
            NodeChild* child = this->root->getNodeChild(role_count, ii);
            if (child->move.equals(last_move)) {
                K273::l_debug("Found next state");
                found_child = child;
                break;
            }
        }

        if (found_child != nullptr) {

            for (int ii=0; ii<this->root->num_children; ii++) {
                NodeChild* child = this->root->getNodeChild(role_count, ii);
                if (child != found_child && child->to_node != nullptr) {

                    Node* next_node = child->to_node;
                    child->to_node = nullptr;

                    ASSERT (next_node->ref_count > 0);
                    next_node->ref_count--;
                    if (next_node->ref_count == 0) {
                        this->releaseNodes(next_node);
                        this->garbage.push_back(next_node);
                    }
                }
            }

            K273::l_error("Garbage collected... %zu, please wait", this->garbage.size());
            for (Node* n : this->garbage) {
                this->removeNode(n);
            }

            this->garbage.clear();

            // may be null
            Node* new_root = found_child->to_node;

            this->root->ref_count--;
            if (this->root->ref_count == 0) {
                K273::l_debug("Removing root node");
                this->removeNode(this->root);

            } else {
                K273::l_debug("What is root ref_count? %d", this->root->ref_count);
            }

            this->root = new_root;

        } else {
            K273::l_error("weird, did not find move in tree root - ref %d", this->root->ref_count);

            this->releaseNodes(this->root);

            K273::l_warning("Garbage collected... %zu please wait", this->garbage.size());
            for (Node* n : this->garbage) {
                this->removeNode(n);
            }

            this->garbage.clear();

            if (this->root->ref_count) {
                this->removeNode(this->root);
            }

            this->root = nullptr;
        }
    }

    K273::l_info("deleted %d nodes", number_of_nodes_before - this->number_of_nodes);
}

std::string Player::beforeApplyInfo() {
    if (this->root != nullptr) {
        NodeChild* best = this->chooseBest(this->root);
        if (best != nullptr && best->to_node != nullptr) {
            return K273::fmtString("nodes=%d, visits=%d, best_prob=%.3f",
                                   this->number_of_nodes,
                                   this->root->visits,
                                   best->to_node->getScore(this->our_role_index));
        }
    }

    return "";
}

int Player::onNextMove(double end_time) {
    const int our_role_index = this->our_role_index;
    if (this->config->skip_single_moves) {
        LegalState* ls = this->sm->getLegalState(our_role_index);
        if (ls->getCount() == 1) {
            int choice = ls->getLegal(0);
            K273::l_info("Only one move - playing it : %s", this->sm->legalToMove(our_role_index, choice));
            return choice;
        }
    }

    double enter_time = get_time();
    K273::l_debug("entering onNextMove() with %.1f seconds", end_time - enter_time);

    if (this->config->max_tree_search_time > 0 && enter_time + this->config->max_tree_search_time < end_time) {
        end_time = enter_time + this->config->max_tree_search_time;
    }

    K273::l_debug("searching for %.1f seconds", end_time - enter_time);

    // create a node for the root (if does not exist already)
    if (this->root == nullptr) {
        K273::l_info("Creating root node");

        Node* n = this->lookupNode(this->sm->getCurrentState());
        if (n != nullptr) {
            K273::l_warning("Refound node!");
            this->root = n;
        } else {
            this->root = this->createNode(this->sm->getCurrentState());
        }

    } else {
        K273::l_info("Root existing with %d nodes", this->number_of_nodes);
    }

    // Great - now start the threads
    ASSERT (this->workers_available.size() == 0);
    ASSERT (this->worker_event_queue.pop() == nullptr);

    for (Worker* worker : this->workers) {
        K273::l_info("Start polling...");
        worker->getThread()->startPolling();
        worker->reset();
        this->workers_available.push(worker);
    }

    K273::l_info("Doing playouts...");

    // times (for debugging)
    this->playout_stats.reset();

    double start_time = get_time();
    double start_time2 = get_rtc_relative_time();
    double next_time = enter_time + this->config->next_time;

    const int role_count = this->sm->getRoleCount();

    const double wide_search_time = (enter_time +
                                     (end_time - enter_time) * this->config->lead_first_node_time_pct);

    bool do_lead_first_node = (role_count != 1 &&
                               this->config->lead_first_node_time_pct > 0 &&
                               this->config->lead_first_node_ucb_constant > 0);

    int playout_next_check_count = 100;
    int last_playouts_since_check_time = 0;
    double last_check_time = get_time();

    while (true) {
        // check elapsed time
        if (this->playout_stats.tree_playouts >= playout_next_check_count) {
            double float_time = get_time();

            // figure out how many playouts done since last time
            int playouts_since_last_time = this->playout_stats.tree_playouts - last_playouts_since_check_time;
            last_playouts_since_check_time = this->playout_stats.tree_playouts;

            // figure out when to test for next
            double playouts_per_second = playouts_since_last_time / (float_time - last_check_time);
            last_check_time = float_time;

            playout_next_check_count = this->playout_stats.tree_playouts + (int) (playouts_per_second * 0.25);
            //K273::l_debug("playouts till next time %d", (int) (playouts_since_last_time * 0.25));

            // bunch of checks, might exit the loop
            if (float_time > end_time) {
                break;

            } else if (this->root->is_finalised) {
                K273::l_warning("Breaking early from tree playouts since root is in terminal state");
                break;

            } else if (float_time > next_time) {
                double average_depth = this->playout_stats.total_tree_playout_depth / (double) this->playout_stats.tree_playouts;

                // get best score for root node
                NodeChild* best = this->chooseBest(this->root, false);
                if (best == nullptr || best->to_node == nullptr) {
                    K273::l_debug("#nodes %d, #playouts %d, av depth: %.2f ---",
                                  this->number_of_nodes,
                                  this->playout_stats.tree_playouts,
                                  average_depth);

                } else {
                    const int our_role_index = this->our_role_index;
                    double our_score = best->to_node->getScore(our_role_index);
                    int choice = best->move.get(our_role_index);
                    K273::l_debug("#nodes %d, #playouts %d, av depth: %.2f, score: %.2f, move: %s",
                                  this->number_of_nodes,
                                  this->playout_stats.tree_playouts,
                                  average_depth,
                                  our_score,
                                  this->sm->legalToMove(our_role_index, choice));
                }

                next_time = float_time + this->config->next_time;
            }

            if (this->playout_stats.tree_playouts > this->config->max_tree_playout_iterations) {
                K273::l_warning("Breaking early since max tree playout iterations.");
                break;
            }

            if (this->node_allocated_memory > this->config->max_memory) {
                K273::l_warning("Breaking since exceeded maximum memory constraint.");
                break;
            }

            if (this->number_of_nodes > this->config->max_number_of_nodes) {
                K273::l_warning("Breaking since exceeded maximum number of nodes.");
                break;
            }

            // expire the do_lead_first_node?
            if (do_lead_first_node) {
                do_lead_first_node = float_time > wide_search_time;
            }
        }

        this->mainLoop(playout_next_check_count, do_lead_first_node);
    }

    K273::l_debug("Done looping.  this->workers_available.size() is %d", (int) this->workers_available.size());

    // Wait for all rollouts to be collected
    this->processAll();

    K273::l_debug("Collected workers.  this->workers_available.size() is %d", (int) this->workers_available.size());

    // stop threads
    for (Worker* worker : this->workers) {
        worker->getThread()->stopPolling();
    }

    // poor mans clear
    this->workers_available = std::queue <Worker*>();

    double total_time_seconds = get_time() - start_time;
    double total_time_seconds2 = get_rtc_relative_time() - start_time2;

    // debug:
    {
        double average_depth = this->playout_stats.total_tree_playout_depth / (double) this->playout_stats.tree_playouts;
        double allocated_megs = this->node_allocated_memory / (1024.0 * 1024.0);
        double av_node_size = this->node_allocated_memory / (double) this->number_of_nodes;

        K273::l_info("------");
        K273::l_info("Nodes %d, memory allocated %.2fM, av %.1f",
                     this->number_of_nodes,
                     allocated_megs,
                     av_node_size);

        // this includes all of the below. Trying to ascertain whether it is OS thing, or what is it making losing % points.
        double pct_main_loop = this->playout_stats.main_loop_accumulative_time / total_time_seconds2;

        double pct_poll = this->playout_stats.poll_available_workers_accumulative_time / total_time_seconds2;
        double pct_search = this->playout_stats.tree_playout_accumulative_time / total_time_seconds2;
        double pct_backprop = this->playout_stats.back_propagate_accumulative_time / total_time_seconds2;
        double pct_expansions = this->playout_stats.expansions_accumulative_time / total_time_seconds2;
        double pct_prompt = this->playout_stats.prompt_accumulative_time / total_time_seconds2;

        double pct_other = 1.0 - pct_poll - pct_search - pct_backprop - pct_expansions - pct_prompt;
        double pct_workers_busy = (this->playout_stats.creation_accumulative_time +
                                   this->playout_stats.rollout_accumulative_time) / (this->config->thread_workers * total_time_seconds2);


        double pct_rollout_busy = this->playout_stats.rollout_accumulative_time / (this->config->thread_workers * total_time_seconds2);

        K273::l_info("Did %d (%.1f p/sec) tree-playouts, %d rollouts, av depth %.2f, main_loop_waiting %ld, transpositions %d",
                     this->playout_stats.tree_playouts,
                     this->playout_stats.tree_playouts / total_time_seconds,
                     this->playout_stats.rollouts,
                     average_depth,
                     this->playout_stats.main_loop_waiting,
                     this->playout_stats.transpositions);

        K273::l_info("Pct search:%.2f / poll %.2f / back:%.2f / expand:%.2f  / prompt:%.2f",
                     pct_search,
                     pct_poll,
                     pct_backprop,
                     pct_expansions,
                     pct_prompt);

        K273::l_info("Pct main:%.2f / other:%.2f / workers:%.2f / rollout:%.2f",
                     pct_main_loop,
                     pct_other,
                     pct_workers_busy,
                     pct_rollout_busy);

        K273::l_info("Finalised :%d, opportunist:%d our_role:%d, unselected:%ld",
                     this->playout_stats.finalised_count,
                     this->playout_stats.finalised_count_opportunist,
                     this->playout_stats.finalised_count_our_role,
                     this->playout_stats.total_unselectable_count);

        K273::l_info("------");

        Node* cur = this->root;
        for (int ii=0; ii<this->config->dump_depth; ii++) {
            std::string indent = "";
            for (int jj=ii-1; jj>=0; jj--) {
                if (jj > 0) {
                    indent += "    ";
                } else {
                    indent += ".   ";
                }
            }

            NodeChild* next_winner = this->chooseBest(cur);
            if (next_winner == nullptr) {
                break;
            }

            Node::dumpNode(cur, next_winner, indent, this->sm);
            cur = next_winner->to_node;
            if (cur == nullptr) {
                break;
            }
        }
    }

    // Choose best move from root (with the most visits)
    NodeChild* winner = this->chooseBest(this->root, true);
    ASSERT(winner != nullptr);

    if (winner->to_node != nullptr && winner->to_node->is_finalised) {
        int fw_depth = 0;
        Node* node = this->root;
        while (true) {
            if (!node->is_finalised) {
                fw_depth = -1;
                break;
            }

            if (node->num_children == 0) {
                break;
            }

            NodeChild* next_winner = this->chooseBest(node, true);
            if (next_winner != nullptr && next_winner->to_node != nullptr) {
                fw_depth++;
                node = next_winner->to_node;

            } else {
                fw_depth = -1;
                break;
            }
        }

        if (fw_depth > 0) {
            K273::l_warning("game finalised in %d moves", fw_depth);
            K273::l_info("Our score: %.2f ", this->root->getScore(this->our_role_index));
        }
    }


    // and return choice
    int choice = winner->move.get(our_role_index);
    K273::l_info("Selected: %s", this->sm->legalToMove(our_role_index, choice));
    return choice;
}
