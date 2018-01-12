#pragma once

#include "puct/node.h"
#include "puct/config.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>
#include <statemachine/jointmove.h>

#include <k273/rng.h>

#include <vector>


namespace GGPZero {

    // forwards
    class NetworkScheduler;

    class PuctEvaluator {
    public:
        PuctEvaluator(const PuctConfig* conf, NetworkScheduler* scheduler);
        virtual ~PuctEvaluator();

    public:
        void updateConf(const PuctConfig* conf);

    private:
        void addNode(PuctNode* new_node);
        void removeNode(PuctNode* n);

        void expandChild(PuctNode* parent, PuctNodeChild* child);
        PuctNode* createNode(const GGPLib::BaseState* state);

        // set dirichlet noise on node
        bool setDirichletNoise(int depth);
        float getPuctConstant(PuctNode* node) const;

    public:
        void updateNodePolicy(PuctNode* node, float* array);
        // do_predictions

        PuctNodeChild* selectChild(PuctNode* node, int depth);

        void backPropagate(float* new_scores);
        int treePlayout();
        void playoutLoop(int max_iterations, double end_time);

        void reset();
        const PuctNode* fastApplyMove(const PuctNodeChild* next);

        const PuctNode* establishRoot(const GGPLib::BaseState* current_state, int game_depth);
        const PuctNodeChild* onNextMove(int max_iterations, double end_time=-1);
        void applyMove(const GGPLib::JointMove* move);

        const PuctNodeChild* choose(const PuctNode* node=nullptr);
        const PuctNodeChild* chooseTopVisits(const PuctNode* node);
        const PuctNodeChild* chooseTemperature(const PuctNode* node);

        std::vector <const PuctNodeChild*> getProbabilities(PuctNode* node, float temperature);

        void logDebug();

        bool hasRoot() const {
            return this->root != nullptr;
        }

    private:
        const PuctConfig* conf;
        NetworkScheduler* scheduler;

        int role_count;
        std::string identifier;

        int game_depth;

        // tree for the entire game
        PuctNode* initial_root;

        // not const PuctNodeChild, as we may need to fix tree
        std::vector <PuctNodeChild*> moves;

        // root for evaluation
        PuctNode* root;
        int number_of_nodes;
        long node_allocated_memory;

        std::vector <PuctNode*> path;

        // random number generator
        xoroshiro64plus32 rng;
    };

}
