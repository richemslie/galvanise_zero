#pragma once

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/rng.h>

#include <vector>

namespace GGPZero {
    // forwards
    class Sample;
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
        SelfPlay(NetworkScheduler* scheduler, const SelfPlayConfig* conf,
                 SelfPlayManager* manager, const GGPLib::BaseState* initial_state);
        ~SelfPlay();

    public:
        void playOnce();
        void playGamesForever();

    private:
        NetworkScheduler* scheduler;
        const SelfPlayConfig* conf;
        SelfPlayManager* manager;
        const GGPLib::BaseState* initial_state;

        // only one evaluator -  allow to swap in/out config
        PuctEvaluator* pe;

        // random number generator
        K273::xoroshiro128plus32 rng;
    };

}
