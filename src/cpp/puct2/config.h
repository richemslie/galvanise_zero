#pragma once

#include <string>


namespace GGPZero::PuctV2 {

    // XXX remove choose_converge_check, or at least something different for pondering
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

        static PuctConfig* defaultConfig();

    };

}
