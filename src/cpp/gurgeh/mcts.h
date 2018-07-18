#pragma once

#include "worker.h"

#include <player/player.h>
#include <player/node.h>
#include <player/path.h>

#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>

#include <k273/rng.h>

#include <queue>
#include <string>
#include <vector>

namespace PlayerMcts {

    // forward
    class Worker;

    class Player : public GGPLib::PlayerBase {
    public:
        Player(GGPLib::StateMachineInterface*, int player_role_index, const Config*);
        virtual ~Player();

    private:
        GGPLib::Node* lookupNode(const GGPLib::BaseState* bs);
        GGPLib::Node* createNode(const GGPLib::BaseState* bs);
        void removeNode(GGPLib::Node* n);
        void releaseNodes(GGPLib::Node* current);

        bool selectChild(GGPLib::Node* node, GGPLib::Path::Selected& path, int depth);
        bool selectChildAdjust(GGPLib::Node* node, GGPLib::Path::Selected& path, int depth);

        void processAll();
        Worker* processAny(bool return_worker);
        void expandTree(Worker* worker);

        void backPropagate(GGPLib::Path::Selected& path, std::vector<double>& new_scores);
        int treePlayout(GGPLib::Path::Selected& path, bool look_wide_at_start);

        GGPLib::NodeChild* chooseBest(GGPLib::Node* node, bool warn=false);
        void mainLoop(int playout_next_check_count, bool do_lead_first_node);

    public:
        // interface:
        virtual void onMetaGaming(double end_time);
        virtual std::string beforeApplyInfo();
        virtual void onApplyMove(GGPLib::JointMove* move);
        virtual int onNextMove(double end_time);

    private:
        const Config* config;

        // tree stuff
        GGPLib::Node* root;
        int number_of_nodes;
        long node_allocated_memory;
        GGPLib::BaseState::HashMap<GGPLib::Node*> lookup;
        std::vector<GGPLib::Node*> garbage;

        std::vector <Worker*> workers;
        std::queue <Worker*> workers_available;

        WorkerEventQueue worker_event_queue;

        struct PlayoutStats {
            double main_loop_accumulative_time;
            long main_loop_waiting;

            double poll_available_workers_accumulative_time;
            double tree_playout_accumulative_time;
            double back_propagate_accumulative_time;
            double expansions_accumulative_time;
            double prompt_accumulative_time;

            // these accumulated from woerks
            double rollout_accumulative_time;
            double creation_accumulative_time;

            int transpositions;
            int total_tree_playout_depth;
            int rollouts;
            int tree_playouts;

            int finalised_count;
            int finalised_count_our_role;
            int finalised_count_opportunist;

            long total_unselectable_count;
            void reset() {
                memset(this, 0, sizeof(PlayoutStats));
            }
        };

        PlayoutStats playout_stats;

        // random number generator
        K273::xoroshiro128plus32 rng;
    };

}
