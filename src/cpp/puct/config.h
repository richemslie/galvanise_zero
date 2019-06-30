#pragma once


namespace GGPZero {


    enum class ChooseFn {
        choose_top_visits, choose_temperature
    };

    struct PuctConfig {
        bool verbose;

        float puct_constant;
        float puct_constant_root;

        float dirichlet_noise_pct;
        float noise_policy_squash_pct;
        float noise_policy_squash_prob;

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

        // < 0, off
        float top_visits_best_guess_converge_ratio;

        float think_time;
        int converged_visits;

        int batch_size;

        // <= 0, off (XXX unused currently)
        int use_legals_count_draw;

        // extra exploration
        float extra_uct_exploration;

        // MCTS prover
        bool backup_finalised;

        // turn on transposition
        bool lookup_transpositions;
    };

}
