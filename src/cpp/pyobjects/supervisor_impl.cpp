#pragma once

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

#include <type_traits>
#include <typeinfo>
#include <cxxabi.h>

template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

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

///////////////////////////////////////////////////////////////////////////////

static PyObject* Supervisor_start_self_play_test(PyObject_Supervisor* self, PyObject* args) {
    int num_selfplays = 0;
    int base_iterations = 0;
    int sample_iterations = 0;

    // sm, transformer, batch_size, expected_policy_size, role_1_index
    if (! ::PyArg_ParseTuple(args, "iii",
                             &num_selfplays, &base_iterations, &sample_iterations)) {
        return nullptr;
    }

    self->impl->selfPlayTest(num_selfplays, base_iterations, sample_iterations);

    Py_RETURN_NONE;
}

static PyObject* Supervisor_fetch_samples(PyObject_Supervisor* self, PyObject* args) {
    // fetch samples
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

    K273::l_warning("type of asFloat is %s", type_name<decltype(asFloat)>().c_str());

    PuctConfig* conf = new PuctConfig;
    conf->name = asString("name");
    conf->verbose = asInt("verbose");
    conf->generation = asString("generation");
    conf->puct_before_expansions = asInt("puct_before_expansions");
    conf->puct_before_root_expansions = asInt("puct_before_root_expansions");
    conf->puct_constant_before = asFloat("puct_constant_before");
    conf->puct_constant_after = asFloat("puct_constant_after");
    conf->dirichlet_noise_pct = asFloat("dirichlet_noise_pct");
    conf->dirichlet_noise_alpha = asFloat("dirichlet_noise_alpha");
    conf->max_dump_depth = asInt("max_dump_depth");
    conf->random_scale = asFloat("random_scale");
    conf->temperature = asFloat("temperature");
    conf->depth_temperature_start = asInt("depth_temperature_start");
    conf->depth_temperature_increment = asFloat("depth_temperature_increment");
    conf->depth_temperature_stop = asInt("depth_temperature_stop");

    std::string choose_method = asString("choose");
    if (choose_method == "choose_top_visits") {
        conf->choose = ChooseFn::choose_top_visits;

    } else if (choose_method == "choose_temperature") {
        conf->choose = ChooseFn::choose_temperature;

    } else {
        K273::l_error("Choose method unknown: '%s', setting to top visits", choose_method.c_str());
        conf->choose = ChooseFn::choose_top_visits;
    }

    self->impl->puctPlayerStart(conf);

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
    {"start_self_play_test", (PyCFunction) Supervisor_start_self_play_test, METH_VARARGS, "start_self_play_test"},
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
