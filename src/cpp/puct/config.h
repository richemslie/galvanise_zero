#pragma once

#include <string>


namespace GGPZero {

    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PuctConfig {
        bool verbose;

        int root_expansions_preset_visits;

        float puct_constant;

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
        float fpu_prior_discount_root;

        float top_visits_best_guess_converge_ratio;
        float evaluation_multipler_to_convergence;

        bool matchmode = false;
    };

}
