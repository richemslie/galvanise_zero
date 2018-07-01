#pragma once


namespace GGPZero::PuctV2 {


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
        // int scaled_visits_at = -1;
        int scaled_visits_at = 1000;
        double scaled_visits_reduce = 4.0;
        double scaled_visits_finalised_reduce = 100.0;

        float max_puct = 3.5f;
        float min_puct = 0.75;
        float min_puct_root = 1.25;

        // < 0, off
        float top_visits_best_guess_converge_ratio = 0.8;
    };


}
