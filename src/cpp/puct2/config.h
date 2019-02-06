#pragma once


namespace GGPZero::PuctV2 {


    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PuctConfig {
        bool verbose;

        float puct_constant = 0.75f;
        float puct_constant_root = 2.5f;
        float puct_multiplier = 1.0f;

        //int puct_before_expansions;
        //int puct_before_root_expansions;

        //int root_expansions_preset_visits;

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

        // < 0, off
        float minimax_backup_ratio = 0.75;
        uint32_t minimax_threshold_visits = 200;

        // < 0, off
        float top_visits_best_guess_converge_ratio = 0.8;

        float think_time = 10.0;
        int converge_relaxed = 5000;

        int batch_size = 32;
    };

}
