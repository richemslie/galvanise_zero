#pragma once

#include "puct/node.h"
#include "puct/config.h"

#include "scheduler.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>
#include <statemachine/jointmove.h>

#include <k273/rng.h>

#include <vector>


namespace GGPZero {

    struct PathElement {
        PathElement(PuctNode* node, PuctNodeChild* choice, PuctNodeChild* best);

        PuctNode* node;
        PuctNodeChild* choice;
        PuctNodeChild* best;
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

        void setDirichletNoise(PuctNode* node);
        float priorScore(PuctNode* node, int depth) const;
        void setPuctConstant(PuctNode* node, int depth) const;
        float getTemperature(int depth) const;

        const PuctNodeChild* choose(const PuctNode* node);
        bool converged(int count) const;

        void checkDrawStates(const PuctNode* node, PuctNode* next);

    private:
        // tree manangement
        void removeNode(PuctNode*);
        void releaseNodes(PuctNode*);
        PuctNode* lookupNode(const GGPLib::BaseState* bs, int depth);
        PuctNode* createNode(PuctNode* parent, const GGPLib::BaseState* state);
        PuctNode* expandChild(PuctNode* parent, PuctNodeChild* child);

        PuctNodeChild* selectChild(PuctNode* node, Path& path);

        void backUpMiniMax(float* new_scores, const PathElement& cur);
        void backup(float* new_scores, const Path& path);

        int treePlayout();

        void playoutWorker(int worker_id);
        void playoutMain(int max_evaluations, double end_time);
        void logDebug(const PuctNodeChild* choice_root);

    public:
        void reset(int game_depth);
        PuctNode* fastApplyMove(const PuctNodeChild* next);
        PuctNode* establishRoot(const GGPLib::BaseState* current_state);

        void resetRootNode();
        const PuctNodeChild* onNextMove(int max_evaluations, double end_time=-1);
        void applyMove(const GGPLib::JointMove* move);

        const PuctNodeChild* chooseTopVisits(const PuctNode* node) const;
        const PuctNodeChild* chooseTemperature(const PuctNode* node);

        Children getProbabilities(PuctNode* node, float temperature, bool use_linger=true);
        void dumpNode(const PuctNode* node, const PuctNodeChild* choice) const;

        int nodeCount() const {
            return this->number_of_nodes;
        }

        GGPLib::StateMachineInterface* getSM() const {
            return this->sm;
        }

        const PuctNode* getRootNode() const {
            return this->root;
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

        // tree for the entire game
        PuctNode* initial_root;

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
