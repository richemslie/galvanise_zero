#pragma once

#include "events.h"
#include "player.h"
#include "player2.h"
#include "selfplay.h"
#include "scheduler.h"
#include "supervisor.h"
#include "puct/config.h"
#include "puct/evaluator.h"
#include "puct2/config.h"
#include "puct2/evaluator.h"

// ggplib imports
#include <statemachine/goalless_sm.h>
#include <statemachine/combined.h>
#include <statemachine/statemachine.h>
#include <statemachine/propagate.h>
#include <statemachine/legalstate.h>
#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>

// k273 includes
#include <k273/util.h>
#include <k273/logging.h>
#include <k273/strutils.h>
#include <k273/exception.h>

#include <string>
#include <vector>

using namespace GGPZero;

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

static PuctConfig* createPuctConfig(PyObject* dict) {
    PuctConfig* config = new PuctConfig;

    auto asInt = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return PyInt_AsLong(borrowed);
    };

    auto asString = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return PyString_AsString(borrowed);
    };

    auto asFloat = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return (float) PyFloat_AsDouble(borrowed);
    };

    config->verbose = asInt("verbose");
    config->puct_before_expansions = asInt("puct_before_expansions");
    config->puct_before_root_expansions = asInt("puct_before_root_expansions");
    config->puct_constant_before = asFloat("puct_constant_before");
    config->puct_constant_after = asFloat("puct_constant_after");
    config->root_expansions_preset_visits = asInt("root_expansions_preset_visits");
    config->dirichlet_noise_pct = asFloat("dirichlet_noise_pct");
    config->dirichlet_noise_alpha = asFloat("dirichlet_noise_alpha");
    config->max_dump_depth = asInt("max_dump_depth");

    config->random_scale = asFloat("random_scale");
    config->temperature = asFloat("temperature");
    config->depth_temperature_start = asInt("depth_temperature_start");
    config->depth_temperature_increment = asFloat("depth_temperature_increment");
    config->depth_temperature_stop = asInt("depth_temperature_stop");
    config->depth_temperature_max = asFloat("depth_temperature_max");

    config->fpu_prior_discount = asFloat("fpu_prior_discount");

    std::string choose_method = asString("choose");
    if (choose_method == "choose_top_visits") {
        config->choose = ChooseFn::choose_top_visits;

    } else if (choose_method == "choose_temperature") {
        config->choose = ChooseFn::choose_temperature;

    } else {
        K273::l_error("Choose method unknown: '%s', setting to top visits", choose_method.c_str());
        config->choose = ChooseFn::choose_top_visits;
    }

    return config;
}

static GGPZero::PuctV2::PuctConfig* createPuctConfigV2(PyObject* dict) {
    GGPZero::PuctV2::PuctConfig* config = new GGPZero::PuctV2::PuctConfig;

    auto asInt = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return PyInt_AsLong(borrowed);
    };

    auto asString = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return PyString_AsString(borrowed);
    };

    auto asFloat = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return (float) PyFloat_AsDouble(borrowed);
    };

    config->verbose = asInt("verbose");

    config->puct_constant = asFloat("puct_constant");
    config->puct_constant_root = asFloat("puct_constant_root");

    config->dirichlet_noise_pct = asFloat("dirichlet_noise_pct");
    config->dirichlet_noise_alpha = asFloat("dirichlet_noise_alpha");
    config->max_dump_depth = asInt("max_dump_depth");

    config->random_scale = asFloat("random_scale");
    config->temperature = asFloat("temperature");
    config->depth_temperature_start = asInt("depth_temperature_start");
    config->depth_temperature_increment = asFloat("depth_temperature_increment");
    config->depth_temperature_stop = asInt("depth_temperature_stop");
    config->depth_temperature_max = asFloat("depth_temperature_max");

    config->fpu_prior_discount = asFloat("fpu_prior_discount");

    config->minimax_backup_ratio = asFloat("minimax_backup_ratio");
    config->minimax_threshold_visits = asInt("minimax_threshold_visits");

    config->top_visits_best_guess_converge_ratio = asFloat("top_visits_best_guess_converge_ratio");

    config->think_time = asFloat("think_time");
    config->converge_relaxed = asInt("converge_relaxed");
    config->converge_non_relaxed = asInt("converge_non_relaxed");

    config->batch_size = asInt("batch_size");

    std::string choose_method = asString("choose");
    if (choose_method == "choose_top_visits") {
        config->choose = GGPZero::PuctV2::ChooseFn::choose_top_visits;

    } else if (choose_method == "choose_temperature") {
        config->choose = GGPZero::PuctV2::ChooseFn::choose_temperature;

    } else {
        K273::l_error("Choose method unknown: '%s', setting to top visits", choose_method.c_str());
        config->choose = GGPZero::PuctV2::ChooseFn::choose_top_visits;
    }

    return config;
}


static SelfPlayConfig* createSelfPlayConfig(PyObject* dict) {
    SelfPlayConfig* config = new SelfPlayConfig;

    auto asInt = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return PyInt_AsLong(borrowed);
    };

    auto asFloat = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        return (float) PyFloat_AsDouble(borrowed);
    };

    auto asDict = [dict] (const char* name) {
        PyObject* borrowed = PyDict_GetItemString(dict, name);
        ASSERT(PyDict_Check(borrowed));
        return borrowed;
    };

    config->max_number_of_samples = asInt("max_number_of_samples");

    config->resign0_score_probability = asFloat("resign0_score_probability");
    config->resign0_false_positive_retry_percentage = asFloat("resign0_false_positive_retry_percentage");

    config->resign1_score_probability = asFloat("resign1_score_probability");
    config->resign1_false_positive_retry_percentage = asFloat("resign1_false_positive_retry_percentage");

    config->select_puct_config = ::createPuctConfig(asDict("select_puct_config"));
    config->select_iterations = asInt("select_iterations");

    config->sample_puct_config = ::createPuctConfig(asDict("sample_puct_config"));
    config->sample_iterations = asInt("sample_iterations");

    config->score_puct_config = ::createPuctConfig(asDict("score_puct_config"));
    config->score_iterations = asInt("score_iterations");
    config->sample_to_end_pct = asFloat("sample_to_end_pct");

    config->abort_max_length = asInt("abort_max_length");
    config->number_repeat_states_draw = asInt("number_repeat_states_draw");
    config->repeat_states_score = asFloat("repeat_states_score");

    config->pct_actually_resign = asFloat("pct_actually_resign");
    config->run_to_end_early_pct = asFloat("run_to_end_early_pct");
    config->run_to_end_early_score = asFloat("run_to_end_early_score");
    config->run_to_end_minimum_game_depth = asInt("run_to_end_minimum_game_depth");

    return config;
}


template <typename T>
static PyObject* doPoll(T* parent_caller, PyObject* args) {
    // IMPORTANT NOTE:
    // the first time around there are no predictions.  In this case two matrices are still passed
    // in - however these are empty arrays.
    // This is to
    //   (a) simplify parse args here.
    //   (b) start the ball rolling, since in the beginning there are no predictions needed to
    //       made, and only once we poll for the first time - then there may be some predictions
    //       to be made!

    // get a list of policies

    int predict_count;
    PyObject* predictions = nullptr;
    if (!PyArg_ParseTuple(args, "iO!", &predict_count, &PyList_Type, &predictions)) {
        return nullptr;
    }

    //K273::l_verbose("# predictions %d, sizeof data %d", predict_count,
    //                (int) PyList_Size(predictions));

    std::vector <float*> data;
    for (int ii=0; ii<PyList_Size(predictions); ii++) {
        PyArrayObject* array = (PyArrayObject*) PyList_GET_ITEM(predictions, ii);

        if (!PyArray_Check(array)) {
            return nullptr;
        }

        if (!PyArray_ISFLOAT(array)) {
            return nullptr;
        }

        if (!PyArray_ISCARRAY(array)) {
            return nullptr;
        }

        data.push_back((float*) PyArray_DATA(array));
    }

    try {
        const ReadyEvent* event = parent_caller->poll(predict_count, data);

        if (event->buf_count) {
            // create a 1D numpy array using our internal array.  It will be resized approriately in python.
            // will check it is of right dimensions/size in python
            npy_intp dims[1]{event->buf_count};
            return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, event->channel_buf);
        }

        // indicates we are done
        Py_RETURN_NONE;

    } catch (...) {
        logExceptionWrapper(__PRETTY_FUNCTION__);
        return nullptr;
    }
}
