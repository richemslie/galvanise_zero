#pragma once

#include "puct2/node.h"
#include "puct2/config.h"

#include "scheduler2.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>
#include <statemachine/jointmove.h>

#include <k273/rng.h>

#include <vector>


namespace GGPZero::PuctV2 {

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
                      NetworkScheduler* scheduler, const GGPZero::GdlBasesTransformer* transformer);
        virtual ~PuctEvaluator();

    public:
        void updateConf(const PuctConfig* conf, const ExtraPuctConfig* extra=nullptr);

    private:
        // tree manangement
        void removeNode(PuctNode*);
        void releaseNodes(PuctNode*);
        PuctNode* lookupNode(const GGPLib::BaseState* bs);
        PuctNode* createNode(PuctNode* parent, const GGPLib::BaseState* state);
        PuctNode* expandChild(PuctNode* parent, PuctNodeChild* child);


        // set dirichlet noise on node
        bool setDirichletNoise(int depth);
        float getPuctConstant(PuctNode* node, int depth) const;

    public:
        PuctNodeChild* selectChild(PuctNode* node, int depth);

        void backUpMiniMax(float* new_node, const PathElement* prev, const PathElement& cur);
        void backPropagate(float* new_scores);
        int treePlayout();
        void playoutLoop(int max_evaluations, double end_time, bool main=false);

        void reset(int game_depth);
        PuctNode* fastApplyMove(const PuctNodeChild* next);
        PuctNode* establishRoot(const GGPLib::BaseState* current_state);

        const PuctNodeChild* onNextMove(int max_evaluations, double end_time=-1);
        void applyMove(const GGPLib::JointMove* move);

        float getTemperature() const;

        const PuctNodeChild* choose(const PuctNode* node=nullptr);
        bool converged(const PuctNode* node) const;
        const PuctNodeChild* chooseTopVisits(const PuctNode* node);
        const PuctNodeChild* chooseTemperature(const PuctNode* node);

        Children getProbabilities(PuctNode* node, float temperature, bool use_linger=true);

        void logDebug(const PuctNodeChild* choice_root);

    private:
        // statemachine shared between evaluators - be careful with its use
        GGPLib::StateMachineInterface* sm;
        GGPLib::BaseState* basestate_expand_node;

        const PuctConfig* conf;
        const ExtraPuctConfig* extra;

        NetworkScheduler* scheduler;

        std::string identifier;

        int game_depth;
        int evaluations;
        int transpositions;

        GGPLib::BaseState::HashMapMasked <PuctNode*>* lookup;

        std::vector <PuctNode*> garbage;

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
