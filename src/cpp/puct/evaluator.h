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
    class SupervisorBase;

    class PuctEvaluator {
    public:
        PuctEvaluator(PuctConfig*, SupervisorBase* supervisor);
        virtual ~PuctEvaluator();

    private:
        void addNode(PuctNode* new_node);
        void removeNode(PuctNode* n);

        void expandChild(PuctNode* parent, PuctNodeChild* child);
        PuctNode* createNode(const GGPLib::BaseState* state);

        // set dirichlet noise on node
        bool setDirichletNoise(int depth);
        double getPuctConstant(PuctNode* node) const;

    public:
        void updateNodePolicy(PuctNode* node, float* array);
        // do_predictions

        PuctNodeChild* selectChild(PuctNode* node, int depth);

        void backPropagate(double* new_scores);
        int treePlayout();
        void playoutLoop(int max_iterations, double end_time);

        void reset();
        PuctNode* fastApplyMove(PuctNodeChild* next);

        PuctNode* establishRoot(const GGPLib::BaseState* current_state, int game_depth);
        PuctNodeChild* onNextMove(int max_iterations, double end_time=-1);
        void applyMove(const GGPLib::JointMove* move);

        void logDebug();

        PuctNodeChild* chooseTopVisits(PuctNode* node=nullptr);
        PuctNodeChild* chooseTemperature(PuctNode* node=nullptr);

        bool hasRoot() const {
            return this->root != nullptr;
        }

    private:
        PuctConfig* config;
        SupervisorBase* supervisor;

        int role_count;
        std::string identifier;

        int game_depth;

        // tree stuff
        PuctNode* root;
        int number_of_nodes;
        long node_allocated_memory;

        std::vector <PuctNode*> path;

        // random number generator
        xoroshiro64plus32 random;
    };

}
