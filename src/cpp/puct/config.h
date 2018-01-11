#pragma once

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

}
