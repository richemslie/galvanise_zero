#pragma once

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/rng.h>

#include <vector>

namespace GGPZero {
    // forwards
    struct Sample;
    struct PuctNode;
    struct PuctConfig;
    class PuctEvaluator;
    class NetworkScheduler;
    class SelfPlayManager;

    struct SelfPlayConfig {
        float oscillate_sampling_pct;

        float temperature_for_policy;

        PuctConfig* puct_config;
        int evals_per_move;

        float resign0_score_probability;
        float resign0_pct;
        float resign1_score_probability;
        float resign1_pct;

        int abort_max_length;
        int number_repeat_states_draw;
        float repeat_states_score;

        float run_to_end_pct;
        int run_to_end_evals;
        PuctConfig* run_to_end_puct_config;
        float run_to_end_early_score;
        int run_to_end_minimum_game_depth;
    };

    class SelfPlay {
    public:
        SelfPlay(SelfPlayManager* manager, const SelfPlayConfig* conf,
                 PuctEvaluator* pe, const GGPLib::BaseState* initial_state,
                 int role_count, std::string identifier);
        ~SelfPlay();

    private:
        bool resign(const PuctNode* node);
        PuctNode* collectSamples(PuctNode* node);
        int runToEnd(PuctNode* node, std::vector <float>& final_scores);
        void addSamples(const std::vector <float>& final_scores,
                        int starting_sample_depth, int game_depth);

        bool checkFalsePositive(const std::vector <float>& false_positive_check_scores,
                                float resign_probability, float final_score,
                                int role_index);

    public:
        void playOnce();
        void playGamesForever();

    private:
        SelfPlayManager* manager;
        const SelfPlayConfig* conf;

        // only one evaluator - allow to swap in/out config
        PuctEvaluator* pe;

        const GGPLib::BaseState* initial_state;
        const int role_count;
        const std::string identifier;

        int match_count;

        // collect samples per game - need to be scored at the end of game
        std::vector <Sample*> game_samples;

        // resignation during self play
        bool has_resigned;
        bool can_resign0;
        bool can_resign1;

        std::vector <float> resign0_false_positive_check_scores;
        std::vector <float> resign1_false_positive_check_scores;

        // random number generator
        K273::xoroshiro128plus32 rng;
    };

}
