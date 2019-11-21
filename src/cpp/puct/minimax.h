#pragma once

#include "puct/node.h"
#include "puct/config.h"
#include "puct/evaluator.h"

#include <statemachine/statemachine.h>

#include <k273/rng.h>

#include <vector>


namespace GGPZero {

struct Specifier {
    int add_policy_count;
    int add_random_count;
};

struct MiniMaxConfig {
    bool verbose;

    float random_scale;
    float temperature;
    int depth_temperature_stop;

    // if the legals <= follow_max, add them in without affecting the minimax.
    int follow_max;

    std::vector<Specifier> minimax_specifier;
};

class MiniMaxer {
  public:
    MiniMaxer(const MiniMaxConfig* conf,
              PuctEvaluator* evaluator,
              GGPLib::StateMachineInterface* sm) :
        conf(conf),
        evaluator(evaluator),
        sm(sm) {
    }

  private:
    PuctNodeChild* minimaxExpanded(PuctNode* node);
    void expandTree(PuctNode* node, int depth);

  private:
    const MiniMaxConfig* conf;
    PuctEvaluator* evaluator;
    GGPLib::StateMachineInterface* sm;

    // random number generator
    K273::xoroshiro128plus32 rng;
};

}  // namespace GGPZero
