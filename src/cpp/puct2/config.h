#pragma once


namespace GGPZero::PuctV2 {


    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PuctConfig {
        bool verbose;

        float puct_constant_init = 0.85f;
        float puct_constant_min = 0.75f;
        float puct_constant_max = 3.5f;
        float puct_constant_min_root = 2.5f;
        float puct_constant_max_root = 5.0f;

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
        int scaled_visits_at = 1000;
        float scaled_visits_reduce = 4.0;
        float scaled_visits_finalised_reduce = 100.0;

        // < 0, off
        float minimax_backup_ratio = 0.75;
        uint32_t minimax_required_visits = 200;

        // < 0, off
        float top_visits_best_guess_converge_ratio = 0.8;

        float think_time = 10.0;
        int converge_relaxed = 5000;
        int converge_non_relaxed = 1000;

        uint32_t expand_threshold_visits = 42;
        int number_of_expansions_end_game = 2;

        int batch_size = 32;

        int policy_dilution_visits = -1;

        // -1 is off
        int root_node_normalisation_limit = 1000;
    };

}
