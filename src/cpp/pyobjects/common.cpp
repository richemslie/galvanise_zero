#pragma once

#include "events.h"
#include "player.h"
#include "selfplay.h"
#include "scheduler.h"
#include "supervisor.h"
#include "puct/config.h"
#include "puct/evaluator.h"

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

    config->name = asString("name");
    config->verbose = asInt("verbose");
    config->generation = asString("generation");
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
    config->depth_temperature_max = asInt("depth_temperature_max");

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

    config->resign_score_probability = asFloat("resign_score_probability");
    config->resign_false_positive_retry_percentage = asFloat("resign_false_positive_retry_percentage");

    config->select_puct_config = ::createPuctConfig(asDict("select_puct_config"));
    config->select_iterations = asInt("select_iterations");

    config->sample_puct_config = ::createPuctConfig(asDict("sample_puct_config"));
    config->sample_iterations = asInt("sample_iterations");

    config->score_puct_config = ::createPuctConfig(asDict("score_puct_config"));
    config->score_iterations = asInt("score_iterations");

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

    PyArrayObject* m0 = nullptr;
    PyArrayObject* m1 = nullptr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &m0, &PyArray_Type, &m1)) {
        return nullptr;
    }

    if (!PyArray_ISFLOAT(m0)) {
        return nullptr;
    }

    if (!PyArray_ISFLOAT(m1)) {
        return nullptr;
    }

    if (!PyArray_ISCARRAY(m0)) {
        return nullptr;
    }

    if (!PyArray_ISCARRAY(m1)) {
        return nullptr;
    }

    int pred_count = PyArray_DIM(m0, 0);
    float* policies = nullptr;
    float* final_scores = nullptr;

    policies = (float*) PyArray_DATA(m0);
    final_scores = (float*) PyArray_DATA(m1);

    try {
        const ReadyEvent* event = parent_caller->poll(policies, final_scores, pred_count);

        if (event->pred_count) {
            // create a numpy array using our internal array
            npy_intp dims[1]{event->pred_count};
            return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, event->channel_buf);
        }

        // indicates we are done
        Py_RETURN_NONE;

    } catch (...) {
        logExceptionWrapper(__PRETTY_FUNCTION__);
        return nullptr;
    }
}
