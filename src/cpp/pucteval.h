#pragma once

#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/statemachine.h>
#include <player/path.h>

#include "puctnode.h"

namespace GGPZero {

    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PUCTEvalConfig {
        std::string name;
        bool vebose;
        std::string generation;

        int playouts_per_iteration;
        int playouts_per_iteration_noop;

        int puct_before_expansions;
        int puct_before_root_expansions;

        double puct_constant_before;
        double puct_constant_after;

        double dirichlet_noise_pct;
        double dirichlet_noise_alpha;

        int expand_root;
        ChooseFn choose;
        int max_dump_depth;

        double random_scale;
        double temperature;
        int depth_temperature_start;
        double depth_temperature_increment;
        int depth_temperature_stop;


        // these are indexes into policy array
        int role0_start_index;
        int role1_start_index;

        // these are passed in
        int role0_noop_legal;
        int role1_noop_legal;

    };

    class PUCTEvaluator {
    public:
        PUCTEvaluator(GGPLib::StateMachineInterface* sm, PUCTEvalConfig* config);
        virtual ~PUCTEvaluator();

    private:
        PuctNode* createNode(const GGPLib::BaseState* bs);
        void removeNode(PuctNode* n);

        void expandChild(PuctNodeChild* child);

        // set dirichlet noise on node
        void setDirichletNoise(PuctNode* node, int depth);
        double getPuctConstant(PuctNode* node) const;

    public:
        void updateNodePolicy(PuctNode* node, float* array);
        // do_predictions

        void selectChild(PuctNode* node, int depth);

        void backPropagate(GGPLib::Path::Selected& path,
                           double* new_scores);
        int treePlayout(PuctNode* node);
        void playoutLoop(PuctNode* node, int max_iterations);

        void logDebug(double total_time_seconds);
        void establishRoot(GGPLib::BaseState* current_state, int game_depth);
        void advanceMove(PuctNodeChild* next);

        PuctNodeChild* chooseTopVisits();
        PuctNodeChild* chooseTemperature();

    private:
        GGPLib::StateMachineInterface* sm;
        PUCTEvalConfig* conf;
        std::string identifier;

        int game_depth;

        // XXX might not need all these
        GGPLib::JointMove* joint_move;
        GGPLib::BaseState* basestate_expand_node;
        GGPLib::BaseState* basestate_expanded_node;

        // tree stuff
        PuctNode* root;
        int number_of_nodes;
        long node_allocated_memory;

        GGPLib::Path::Selected path;

        // XXX better random K273::Random random;
    };

}
