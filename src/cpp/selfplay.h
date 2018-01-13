#pragma once

#include <statemachine/statemachine.h>

#include <k273/rng.h>

#include <vector>

/*
 * never add a reference back to supervisor... the supervisor will always call us
 */


namespace GGPZero {
    // forwards
    class Sample;
    class PuctConfig;
    class PuctEvaluator;
    class NetworkScheduler;

    struct SelfPlayConfig {
        // choose a random number between 0 - expected_game_length for samples to start
        int expected_game_length;

        // a node score reaches probability of winning, start sample selection early
        float early_sample_start_probability;

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
                 GGPLib::StateMachineInterface* sm);
        ~SelfPlay();

    public:
        void playOnce();
        void playGamesForever();

        std::vector <Sample*>& getSamples() {
            return samples;
        }

    private:
        NetworkScheduler* scheduler;
        const SelfPlayConfig* conf;

        // only one evaluator -  allow to swap in/out config
        PuctEvaluator* pe;

        const GGPLib::BaseState* initial_state;
        std::vector <Sample*> samples;

        // random number generator
        xoroshiro64plus32 rng;
    };

}
