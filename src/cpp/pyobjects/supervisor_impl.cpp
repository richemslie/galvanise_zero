#pragma once

#include "sample.h"
#include "selfplay.h"
#include "supervisor.h"

// k273 includes
#include <k273/util.h>
#include <k273/logging.h>
#include <k273/strutils.h>
#include <k273/exception.h>

// ggplib imports
#include <statemachine/goalless_sm.h>
#include <statemachine/combined.h>
#include <statemachine/statemachine.h>
#include <statemachine/propagate.h>
#include <statemachine/legalstate.h>
#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>

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

///////////////////////////////////////////////////////////////////////////////

struct PyObject_Supervisor {
    PyObject_HEAD
    Supervisor* impl;
};

PuctConfig* createPuctConfig(PyObject* dict) {
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
    config->dirichlet_noise_pct = asFloat("dirichlet_noise_pct");
    config->dirichlet_noise_alpha = asFloat("dirichlet_noise_alpha");
    config->max_dump_depth = asInt("max_dump_depth");
    config->random_scale = asFloat("random_scale");
    config->temperature = asFloat("temperature");
    config->depth_temperature_start = asInt("depth_temperature_start");
    config->depth_temperature_increment = asFloat("depth_temperature_increment");
    config->depth_temperature_stop = asInt("depth_temperature_stop");

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

SelfPlayConfig* createSelfPlayConfig(PyObject* dict) {
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

    config->expected_game_length = asInt("expected_game_length");
    config->early_sample_start_probability = asFloat("early_sample_start_probability");
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

///////////////////////////////////////////////////////////////////////////////

static PyObject* Supervisor_start_self_play(PyObject_Supervisor* self, PyObject* args) {
    PyObject* dict;
    if (! PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        return nullptr;
    }

    SelfPlayConfig* config = ::createSelfPlayConfig(dict);
    self->impl->startSelfPlay(config);

    Py_RETURN_NONE;
}

PyObject* stateToList(const GGPLib::BaseState* bs) {
    // ref counts all stolen :)
    PyObject* state_as_list = PyList_New(bs->size);
    for (int ii=0; ii<bs->size; ii++) {
        PyList_SetItem(state_as_list, ii, PyInt_FromLong(bs->get(ii)));
    }

    return state_as_list;
}

PyObject* policyElementToTuple(std::pair<int, float>& e) {
    // ref counts all stolen :)
    PyObject* tup = PyTuple_New(2);
    PyTuple_SetItem(tup, 0, PyInt_FromLong(e.first));
    PyTuple_SetItem(tup, 1, PyFloat_FromDouble((double) e.second));
    return tup;
}

void PyDict_setNewItem(PyObject* d, const char* name, PyObject* o) {
    /* steals ref */
    ASSERT(PyDict_SetItemString(d, name, o) == 0);
    Py_DECREF(o);
}

PyObject* sampleToDict(Sample* sample) {
    // this is basically FML - trying to steal as much ref counts as possible
    PyObject* sample_as_dict = PyDict_New();

    // easy one first
    PyDict_setNewItem(sample_as_dict, "state", stateToList(sample->state));
    PyDict_setNewItem(sample_as_dict, "depth", PyInt_FromLong(sample->depth));
    PyDict_setNewItem(sample_as_dict, "game_length", PyInt_FromLong(sample->game_length));
    PyDict_setNewItem(sample_as_dict, "lead_role_index", PyInt_FromLong(sample->lead_role_index));

    // same pattern, could abstract it out if my template skills were better
    {
        int num_prev_states = sample->prev_states.size();
        PyObject* prev_states = PyList_New(num_prev_states);
        for (int ii=0; ii<num_prev_states; ii++) {
            PyList_SetItem(prev_states, ii, stateToList(sample->prev_states[ii]));
        }
        PyDict_setNewItem(sample_as_dict, "prev_state", prev_states);
    }

    {
        int num_final_score = sample->final_score.size();
        PyObject* final_score = PyList_New(num_final_score);
        for (int ii=0; ii<num_final_score; ii++) {
            PyList_SetItem(final_score, ii, PyFloat_FromDouble((double) sample->final_score[ii]));
        }

        PyDict_setNewItem(sample_as_dict, "final_score", final_score);
    }

    {
        int num_policy = sample->policy.size();
        PyObject* policy = PyList_New(num_policy);
        for (int ii=0; ii<num_policy; ii++) {
            PyList_SetItem(policy, ii, policyElementToTuple(sample->policy[ii]));
        }

        PyDict_setNewItem(sample_as_dict, "policy", policy);
    }

    return sample_as_dict;
}

static PyObject* Supervisor_fetch_samples(PyObject_Supervisor* self, PyObject* args) {
    std::vector <Sample*> samples = self->impl->getSamples();

    if (!samples.empty()) {
        int sample_count = samples.size();
        PyObject* py_samples = PyList_New(sample_count);
        for (int ii=0; ii<sample_count; ii++) {
            PyList_SetItem(py_samples, ii, sampleToDict(samples[ii]));
        }

        return py_samples;
    }

    Py_RETURN_NONE;
}

static PyObject* Supervisor_poll(PyObject_Supervisor* self, PyObject* args) {
    // IMPORTANT NOTE:
    // the first time around there are no predictions.  In this case two matrices are still passed
    // in - however these are empty arrays.
    // This is just to simplify parse args here.

    PyArrayObject* m0 = nullptr;
    PyArrayObject* m1 = nullptr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &m0, &PyArray_Type, &m1)) {
        return nullptr;
    }

    // for (int ii=0; ii<PyArray_NDIM(m0); ii++) {
    //     K273::l_verbose("dim of m0 @ %d : %d", ii, (int) PyArray_DIM(m0, ii));
    // }

    // for (int ii=0; ii<PyArray_NDIM(m1); ii++) {
    //     K273::l_verbose("dim of m1 @ %d : %d", ii, (int) PyArray_DIM(m1, ii));
    // }

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

    int sz = PyArray_DIM(m0, 0);
    // K273::l_verbose("dim of m0 : %d", sz);

    float* policies = nullptr;
    float* final_scores = nullptr;

    if (sz) {
        policies = (float*) PyArray_DATA(m0);
        final_scores = (float*) PyArray_DATA(m1);
    }

    try {
        int res = self->impl->poll(policies, final_scores, sz);

        if (res) {
            // create a numpy array using our internal array
            npy_intp dims[1]{res};
            return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, self->impl->getBuf());

        } else {
            // indicates we are done
            Py_RETURN_NONE;
        }

    } catch (...) {
        logExceptionWrapper(__PRETTY_FUNCTION__);
        return nullptr;
    }
}

static PyObject* Supervisor_player_start(PyObject_Supervisor* self, PyObject* args) {
    PyObject* dict;
    if (! PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        return nullptr;
    }

    PuctConfig* config = createPuctConfig(dict);
    self->impl->puctPlayerStart(config);

    Py_RETURN_NONE;
}

static PyObject* Supervisor_player_reset(PyObject_Supervisor* self, PyObject* args) {
    self->impl->puctPlayerReset();
    Py_RETURN_NONE;
}

static PyObject* Supervisor_player_apply_move(PyObject_Supervisor* self, PyObject* args) {
    ssize_t ptr = 0;
    if (! ::PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }

    GGPLib::JointMove* move = reinterpret_cast<GGPLib::JointMove*> (ptr);
    self->impl->puctApplyMove(move);

    Py_RETURN_NONE;
}

static PyObject* Supervisor_player_move(PyObject_Supervisor* self, PyObject* args) {
    ssize_t ptr = 0;
    int iterations = 0;
    double end_time = 0.0;
    if (! ::PyArg_ParseTuple(args, "nid", &ptr, &iterations, &end_time)) {
        return nullptr;
    }

    GGPLib::BaseState* basestate = reinterpret_cast<GGPLib::BaseState*> (ptr);
    self->impl->puctPlayerMove(basestate, iterations, end_time);
    Py_RETURN_NONE;
}

static PyObject* Supervisor_player_get_move(PyObject_Supervisor* self, PyObject* args) {
    int lead_role_index = 0;
    if (! ::PyArg_ParseTuple(args, "i", &lead_role_index)) {
        return nullptr;
    }

    int res = self->impl->puctPlayerGetMove(lead_role_index);
    return ::Py_BuildValue("i", res);
    Py_RETURN_NONE;
}


static struct PyMethodDef Supervisor_methods[] = {
    {"start_self_play", (PyCFunction) Supervisor_start_self_play, METH_VARARGS, "start_self_play"},
    {"fetch_samples", (PyCFunction) Supervisor_fetch_samples, METH_NOARGS, "fetch_samples"},

    {"poll", (PyCFunction) Supervisor_poll, METH_VARARGS, "poll"},

    {"player_start", (PyCFunction) Supervisor_player_start, METH_VARARGS, "player_start"},
    {"player_reset", (PyCFunction) Supervisor_player_reset, METH_NOARGS, "player_reset"},
    {"player_apply_move", (PyCFunction) Supervisor_player_apply_move, METH_VARARGS, "player_apply_move"},
    {"player_move", (PyCFunction) Supervisor_player_move, METH_VARARGS, "player_move"},
    {"player_get_move", (PyCFunction) Supervisor_player_get_move, METH_VARARGS, "player_get_move"},



    {nullptr, nullptr}            /* Sentinel */
};

static void Supervisor_dealloc(PyObject* ptr);


static PyTypeObject PyType_Supervisor = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "Supervisor",                /*tp_name*/
    sizeof(PyObject_Supervisor), /*tp_size*/
    0,                        /*tp_itemsize*/

    /* methods */
    Supervisor_dealloc,    /*tp_dealloc*/
    0,                  /*tp_print*/
    0,                  /*tp_getattr*/
    0,                  /*tp_setattr*/
    0,                  /*tp_compare*/
    0,                  /*tp_repr*/
    0,                  /*tp_as_number*/
    0,                  /*tp_as_sequence*/
    0,                  /*tp_as_mapping*/
    0,                  /*tp_hash*/
    0,                  /*tp_call*/
    0,                  /*tp_str*/
    0,                  /*tp_getattro*/
    0,                  /*tp_setattro*/
    0,                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    0,                  /*tp_doc*/
    0,                  /*tp_traverse*/
    0,                  /*tp_clear*/
    0,                  /*tp_richcompare*/
    0,                  /*tp_weaklistoffset*/
    0,                  /*tp_iter*/
    0,                  /*tp_iternext*/
    Supervisor_methods,    /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
};

static PyObject_Supervisor* PyType_Supervisor_new(Supervisor* impl) {
    PyObject_Supervisor* res = PyObject_New(PyObject_Supervisor,
                                                  &PyType_Supervisor);
    res->impl = impl;
    return res;
}


static void Supervisor_dealloc(PyObject* ptr) {
    K273::l_debug("--> Supervisor_dealloc");
    ::PyObject_Del(ptr);
}

///////////////////////////////////////////////////////////////////////////////

static PyObject* gi_Supervisor(PyObject* self, PyObject* args) {
    ssize_t ptr = 0;
    PyObject* obj = 0;
    int batch_size = 0;
    int expected_policy_size = 0;
    int role_1_index = 0;

    // sm, transformer, batch_size, expected_policy_size, role_1_index
    if (! ::PyArg_ParseTuple(args, "nO!iii", &ptr,
                             &(PyType_GdlBasesTransformerWrapper), &obj,
                             &batch_size, &expected_policy_size, &role_1_index)) {
        return nullptr;
    }

    PyObject_GdlBasesTransformerWrapper* py_transformer = (PyObject_GdlBasesTransformerWrapper*) obj;

    GGPLib::StateMachine* sm = reinterpret_cast<GGPLib::StateMachine*> (ptr);

    // create the c++ object
    Supervisor* supervisor = new Supervisor(sm);
    supervisor->createScheduler(py_transformer->impl, batch_size,
                                expected_policy_size, role_1_index);

    return (PyObject*) PyType_Supervisor_new(supervisor);
}
