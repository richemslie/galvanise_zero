#pragma once

#include "puct2/node.h"
#include "puct2/config.h"

#include "scheduler.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>
#include <statemachine/jointmove.h>

#include <k273/rng.h>

#include <vector>


namespace GGPZero::PuctV2 {

    struct PathElement {
        PathElement(PuctNode* node, PuctNodeChild* choice, PuctNodeChild* best, int num_expanded_children);

        PuctNode* node;
        PuctNodeChild* choice;
        PuctNodeChild* best;
        int num_children_expanded;
    };

    using Path = std::vector <PathElement>;

    ///////////////////////////////////////////////////////////////////////////////

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
        // note can't be const method as rng state modified
        std::vector <float> getDirichletNoise(int depth);
        void setPuctConstant(PuctNode* node, int depth) const;

    public:
        PuctNodeChild* selectChild(PuctNode* node, Path& path);

        void backUpMiniMax(float* new_node, const PathElement* prev, const PathElement& cur);
        void backPropagate(float* new_scores, const Path& path);
        int treePlayout();

        void playoutLoop(int max_evaluations, double end_time, bool main=false);

        void reset(int game_depth);
        PuctNode* fastApplyMove(const PuctNodeChild* next);
        PuctNode* establishRoot(const GGPLib::BaseState* current_state);

        const PuctNodeChild* onNextMove(int max_evaluations, double end_time=-1);
        void applyMove(const GGPLib::JointMove* move);

        float getTemperature() const;

        const PuctNodeChild* choose(const PuctNode* node=nullptr);
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

        // stats
        int stats_finals;
        int stats_blocked;
        int stats_tree_playouts;
        int stats_transpositions;
        int stats_total_depth;
        int stats_max_depth;

        GGPLib::BaseState::HashMapMasked <PuctNode*>* lookup;

        std::vector <PuctNode*> garbage;
        std::vector <PuctNodeChild*> moves;

        // root for evaluation
        PuctNode* root;
        int number_of_nodes;
        long node_allocated_memory;

        // random number generator
        K273::xoroshiro128plus32 rng;
    };

}
