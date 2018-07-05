#pragma once

#include <string>


namespace GGPZero {

    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PuctConfig {
        bool verbose;

        int puct_before_expansions;
        int puct_before_root_expansions;

        int root_expansions_preset_visits;

        float puct_constant_before;
        float puct_constant_after;

        float dirichlet_noise_pct;
        float dirichlet_noise_alpha;

        ChooseFn choose;
        int max_dump_depth;

        float random_scale;

        float temperature;
        int depth_temperature_start;
        float depth_temperature_increment;
        int depth_temperature_stop;
        float depth_temperature_max;

        float fpu_prior_discount;
    };

    struct ExtraPuctConfig {
        bool matchmode = false;

        // finalised nodes on (only set during backprop, so this turns it on)
        bool backprop_finalised = true;

        // < 0, off
        float top_visits_best_guess_converge_ratio = 0.8;
        float cpuct_after_root_multiplier = 2.0;
        bool bypass_evaluation_single_node = true;

        double evaluation_multipler_on_terminal = 1.5;
        double evaluation_multipler_to_convergence = 2;

        // Moves the score so that PUCT exploration is normalised
        bool adjust_score_puct_normalisation = true;
    };

}
