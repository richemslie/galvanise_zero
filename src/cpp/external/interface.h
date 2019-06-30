#pragma once

#define StateMachine void

#ifdef __cplusplus
extern "C" {
#endif

    // CFFI START INCLUDE
    StateMachine* getSMDraughts_10x10();
    StateMachine* getSMDraughtsKiller_10x10();

    // CFFI END INCLUDE

#ifdef __cplusplus
}
#endif

#undef StateMachine
