#pragma once


namespace GGPZero::PuctV2 {


    enum class ChooseFn {
        choose_top_visits, choose_converge_check, choose_temperature
    };

    struct PuctConfig {
        bool verbose;

        float puct_constant;
        float puct_constant_root;

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

        // < 0, off
        float minimax_backup_ratio;
        uint32_t minimax_threshold_visits;

        // < 0, off
        float top_visits_best_guess_converge_ratio;

        float think_time;
        int converge_relaxed;

        int batch_size;

        // prevent latching at root %.. < 0 off.
        float limit_latch_root = -1;
    };

}
