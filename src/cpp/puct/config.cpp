#include "puct/config.h"

using namespace GGPZero;

PuctConfig* PuctConfig::defaultConfig() {
    PuctConfig* conf = new PuctConfig;
    conf->name = "test_evaluator";
    conf->verbose = false;
    conf->generation = "noone";

    conf->puct_before_expansions = 3;
    conf->puct_before_root_expansions = 5;

    conf->puct_constant_before = 3.0;
    conf->puct_constant_after = 0.75;

    conf->dirichlet_noise_pct = 0.25;
    conf->dirichlet_noise_alpha = 0.1;

    conf->choose = ChooseFn::choose_top_visits;

    conf->max_dump_depth = 3;

    conf->random_scale = 0.75;
    conf->temperature = 1.0;
    conf->depth_temperature_start = 8;
    conf->depth_temperature_stop = 16;
    conf->depth_temperature_increment = 0.5;

    return conf;
}
