#pragma once

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/rng.h>

#include <vector>

namespace GGPZero {
    // forwards
    class Sample;
    class PuctNode;
    class PuctConfig;
    class PuctEvaluator;
    class NetworkScheduler;
    class SelfPlayManager;

    struct SelfPlayConfig {
        // -1 is off, and defaults to alpha-zero style
        int max_number_of_samples;


        PuctConfig* select_puct_config;
        int select_iterations;

        PuctConfig* sample_puct_config;
        int sample_iterations;

        PuctConfig* score_puct_config;
        int score_iterations;

        float resign0_score_probability;
        float resign0_false_positive_retry_percentage;
        float resign1_score_probability;
        float resign1_false_positive_retry_percentage;
    };

    class SelfPlay {
    public:
        SelfPlay(SelfPlayManager* manager, const SelfPlayConfig* conf,
                 PuctEvaluator* pe, const GGPLib::BaseState* initial_state,
                 int role_count, std::string identifier);
        ~SelfPlay();

    private:
        PuctNode* selectNode();
        bool resign(PuctNode* node);
        PuctNode* collectSamples(PuctNode* node);
        PuctNode* runToEnd(PuctNode* node);

        bool checkFalsePositive(const std::vector <float>& false_positive_check_scores,
                                float resign_probability, float final_score,
                                int role_index);

    public:
        void playOnce();
        void playGamesForever();

    private:
        SelfPlayManager* manager;
        const SelfPlayConfig* conf;

        // only one evaluator -  allow to swap in/out config
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
