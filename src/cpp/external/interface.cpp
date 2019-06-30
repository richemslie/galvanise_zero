#include "interface.h"

#include "draughts_desc.h"
#include "draughts_board.h"
#include "draughts_sm.h"

#include <k273/logging.h>
#include <k273/exception.h>

#include <statemachine/statemachine.h>

#include <string>

static void logExceptionWrapper(const std::string& name) {
    try {
        K273::l_critical("an exception was thrown in in %s:", name.c_str());
        throw;

    } catch (const K273::Exception& exc) {
        K273::l_critical("K273::Exception Message : %s", exc.getMessage().c_str());
        K273::l_critical("K273::Exception Stacktrace : \n%s", exc.getStacktrace().c_str());

    } catch (std::exception& exc) {
        K273::l_critical("std::exception What : %s", exc.what());

    } catch (...) {
        K273::l_critical("Unknown exception");
    }
}

void* getSMDraughts_10x10() {
    K273::l_info("in getSMDraughts_10x10()");

    try {
        InternationalDraughts::Description* desc = new InternationalDraughts::Description(10);
        InternationalDraughts::Board* board = new InternationalDraughts::Board(desc);
        GGPLib::StateMachineInterface* sm = new InternationalDraughts::SM(board, desc);

        return (void *) sm;

    } catch (...) {
        logExceptionWrapper(__PRETTY_FUNCTION__);
    }

    return nullptr;
}

void* getSMDraughtsKiller_10x10() {
    K273::l_info("in getSMDraughtsKiller_10x10()");

    try {
        InternationalDraughts::Description* desc = new InternationalDraughts::Description(10);
        InternationalDraughts::Board* board = new InternationalDraughts::Board(desc, false, true);
        GGPLib::StateMachineInterface* sm = new InternationalDraughts::SM(board, desc);

        return (void *) sm;

    } catch (...) {
        logExceptionWrapper(__PRETTY_FUNCTION__);
    }

    return nullptr;
}
