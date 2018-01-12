#pragma once

#include <string>


namespace GGPZero {

    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PuctConfig {
        std::string name;
        bool verbose;
        std::string generation;

        int puct_before_expansions;
        int puct_before_root_expansions;

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

        static PuctConfig* defaultConfig();

    };

}
