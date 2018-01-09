/* Start out as a simple as co-routine implementation.  Optimize later. */

#pragma once

#include "puctnode.h"
#include "rng.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <vector>

namespace GGPZero {

    // forwards
    class Supervisor;

    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PUCTEvalConfig {
        std::string name;
        bool verbose;
        std::string generation;

        int puct_before_expansions;
        int puct_before_root_expansions;

        double puct_constant_before;
        double puct_constant_after;

        double dirichlet_noise_pct;
        double dirichlet_noise_alpha;

        ChooseFn choose;
        int max_dump_depth;

        double random_scale;
        double temperature;
        int depth_temperature_start;
        double depth_temperature_increment;
        int depth_temperature_stop;
    };

    class PuctEvaluator {
    public:
        PuctEvaluator(PUCTEvalConfig*, Supervisor* supervisor);
        virtual ~PuctEvaluator();

    private:
        void addNode(PuctNode* new_node);
        void removeNode(PuctNode* n);

        void expandChild(PuctNode* parent, PuctNodeChild* child);
        PuctNode* createNode(GGPLib::BaseState* state);

        // set dirichlet noise on node
        bool setDirichletNoise(int depth);
        double getPuctConstant(PuctNode* node) const;

    public:
        void updateNodePolicy(PuctNode* node, float* array);
        // do_predictions

        PuctNodeChild* selectChild(PuctNode* node, int depth);

        void backPropagate(double* new_scores);
        int treePlayout();
        void playoutLoop(int max_iterations);

        PuctNode* fastApplyMove(PuctNodeChild* next);
        void reset();

        PuctNode* establishRoot(GGPLib::BaseState* current_state, int game_depth);
        PuctNodeChild* onNextNove(int max_iterations);

        void logDebug(PuctNodeChild* choice);

        PuctNodeChild* chooseTopVisits();
        PuctNodeChild* chooseTemperature();

    private:
        Supervisor* supervisor;
        PUCTEvalConfig* config;
        int role_count;
        std::string identifier;

        int game_depth;

        GGPLib::BaseState* basestate_expand_node;

        // tree stuff
        PuctNode* root;
        int number_of_nodes;
        long node_allocated_memory;

        std::vector <PuctNode*> path;

        // random number generator
        xoroshiro64plus32 random;
    };

}
