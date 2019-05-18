#pragma once

#include "puct/node.h"
#include "puct/config.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>
#include <statemachine/jointmove.h>

#include <k273/rng.h>

#include <vector>


// steps:
// 1. make path same
// 2. add transpositions

namespace GGPZero {

    struct PathElement {
        PathElement(PuctNodeChild* c, PuctNode* n) :
            child(c),
            to_node(n) {
        }

        PuctNodeChild* child;
        PuctNode* to_node;
    };

    // forwards
    class NetworkScheduler;

    class PuctEvaluator {
    public:
        PuctEvaluator(GGPLib::StateMachineInterface* sm, const PuctConfig* conf,
                      NetworkScheduler* scheduler,
                      const GGPZero::GdlBasesTransformer* transformer);
        virtual ~PuctEvaluator();

    public:
        void updateConf(const PuctConfig* conf);

        // special config for self play only
        void setRepeatStateDraw(int number_repeat_states_draw, float repeat_states_score);

    private:
        void addNode(PuctNode* new_node);
        void removeNode(PuctNode* n);

        void expandChild(PuctNode* parent, PuctNodeChild* child, bool expansion_time=false);
        PuctNode* createNode(PuctNode* parent, const GGPLib::BaseState* state, bool expansion_time=false);

        void checkDrawStates(const PuctNode* node, PuctNode* next);

    public:
#include "unify.h"

        // do_predictions
        PuctNodeChild* selectChild(PuctNode* node, int depth);

        void backPropagate(float* new_scores);
        int treePlayout();
        void playoutLoop(int max_evaluations, double end_time);

        void reset(int game_depth);
        PuctNode* fastApplyMove(const PuctNodeChild* next);
        PuctNode* establishRoot(const GGPLib::BaseState* current_state);

        const PuctNodeChild* onNextMove(int max_evaluations, double end_time=-1);
        void applyMove(const GGPLib::JointMove* move);

        const PuctNodeChild* chooseTopVisits(const PuctNode* node) const;
        const PuctNodeChild* chooseTemperature(const PuctNode* node);

        Children getProbabilities(PuctNode* node, float temperature, bool use_policy=true);

        void logDebug(const PuctNodeChild* choice_root);

        int nodeCount() const {
            return this->number_of_nodes;
        }

    private:
        // statemachine shared between evaluators - be careful with its use
        GGPLib::StateMachineInterface* sm;
        GGPLib::BaseState* basestate_expand_node;

        const PuctConfig* conf;

        // introduce a different way of doing things
        int number_repeat_states_draw;
        float repeat_states_score;
        GGPLib::BaseState::EqualsMasked* masked_bs_equals;

        NetworkScheduler* scheduler;

        int game_depth;
        int evaluations;

        // tree for the entire game
        PuctNode* initial_root;

        // not const PuctNodeChild, as we may need to fix tree
        std::vector <PuctNodeChild*> moves;

        // root for evaluation
        PuctNode* root;
        int number_of_nodes;
        long node_allocated_memory;

        std::vector <PathElement> path;

        // random number generator
        K273::xoroshiro128plus32 rng;
    };

}
