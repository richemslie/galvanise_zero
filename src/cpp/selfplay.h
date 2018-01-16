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

        // if the probability of losing drops below - then resign
        float resign_score_probability;

        // ignore resignation - and continue to end
        float resign_false_positive_retry_percentage;

        PuctConfig* select_puct_config;
        int select_iterations;

        PuctConfig* sample_puct_config;
        int sample_iterations;

        PuctConfig* score_puct_config;
        int score_iterations;
    };

    class SelfPlay {
    public:
        SelfPlay(SelfPlayManager* manager, const SelfPlayConfig* conf,
                 PuctEvaluator* pe,
                 const GGPLib::BaseState* initial_state, int role_count);
        ~SelfPlay();


    private:
        PuctNode* selectNode(const bool can_resign);
        PuctNode* collectSamples(PuctNode* node);

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

        // random number generator
        K273::xoroshiro128plus32 rng;

        // local to playOnce();
        std::vector <Sample*> game_samples;
    };

}
