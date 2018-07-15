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
        PuctEvaluator(GGPLib::StateMachineInterface* sm, NetworkScheduler* scheduler,
                      const GGPZero::GdlBasesTransformer* transformer);
        virtual ~PuctEvaluator();

    public:
        // called after creation
        void updateConf(const PuctConfig* conf);

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

        bool converged(int count) const;
        PuctNodeChild* selectChild(PuctNode* node, Path& path);

        void backUpMiniMax(float* new_scores, const PathElement& cur);
        void backPropagate(float* new_scores, const Path& path);

        int treePlayout();

        void normaliseRootNode();
        void playoutWorker();
        void playoutMain(double end_time);
        void logDebug(const PuctNodeChild* choice_root);

    public:
        void reset(int game_depth);
        PuctNode* fastApplyMove(const PuctNodeChild* next);
        PuctNode* establishRoot(const GGPLib::BaseState* current_state);

        const PuctNodeChild* onNextMove(int max_evaluations, double end_time);
        void applyMove(const GGPLib::JointMove* move);

        float getTemperature() const;

        const PuctNodeChild* choose(const PuctNode* node=nullptr);
        const PuctNodeChild* chooseTopVisits(const PuctNode* node);
        const PuctNodeChild* chooseTemperature(const PuctNode* node);

        Children getProbabilities(PuctNode* node, float temperature, bool use_linger=true);

        int nodeCount() const {
            return this->number_of_nodes;
        }

    private:
        struct PlayoutStats {
            PlayoutStats() {
                this->reset();
            }

            void reset() {
                this->num_blocked = 0;
                this->num_tree_playouts = 0;
                this->num_evaluations = 0;
                this->num_transpositions_attached = 0;

                this->playouts_total_depth = 0;
                this->playouts_max_depth = 0;
                this->playouts_finals = 0;
            }

            int num_blocked;
            int num_tree_playouts;
            int num_evaluations;
            int num_transpositions_attached;

            int playouts_total_depth;
            int playouts_max_depth;
            int playouts_finals;
        };

    private:
        const PuctConfig* conf;

        GGPLib::StateMachineInterface* sm;
        GGPLib::BaseState* basestate_expand_node;
        NetworkScheduler* scheduler;

        int game_depth;


        // root of the tree
        PuctNode* root;

        // lookup table to tree
        GGPLib::BaseState::HashMapMasked <PuctNode*>* lookup;

        // when releasing nodes from tree, puts them to delete afterwards
        std::vector <PuctNode*> garbage;

        // tree info
        int number_of_nodes;
        long node_allocated_memory;

        // used by workers to indicate work to do
        bool do_playouts;

        // stats collected during playouts
        PlayoutStats stats;

        // random number generator
        K273::xoroshiro128plus32 rng;
    };

}
