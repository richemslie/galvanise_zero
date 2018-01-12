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

using namespace GGPZero;

///////////////////////////////////////////////////////////////////////////////

struct PyObject_InlineSupervisor {
    PyObject_HEAD
    Supervisor* impl;
};

///////////////////////////////////////////////////////////////////////////////

static PyObject* InlineSupervisor_start_self_play_test(PyObject_InlineSupervisor* self, PyObject* args) {
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

static PyObject* InlineSupervisor_poll(PyObject_InlineSupervisor* self, PyObject* args) {
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

    float* policies = (float*) PyArray_DATA(m0);
    float* final_scores = (float*) PyArray_DATA(m1);

    int res = self->impl->poll(policies, final_scores, sz);

    if (res) {
        // create a numpy array using our internal array
        npy_intp dims[1]{res};
        return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, self->impl->getBuf());

    } else {
        // indicates we are done
        Py_RETURN_NONE;
    }
}


static PyObject* InlineSupervisor_player_start(PyObject_InlineSupervisor* self, PyObject* args) {
    self->impl->puctPlayerStart();
    Py_RETURN_NONE;
}

static PyObject* InlineSupervisor_player_apply_move(PyObject_InlineSupervisor* self, PyObject* args) {
    ssize_t ptr = 0;
    if (! ::PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }

    GGPLib::JointMove* move = reinterpret_cast<GGPLib::JointMove*> (ptr);
    self->impl->puctApplyMove(move);

    Py_RETURN_NONE;
}

static PyObject* InlineSupervisor_player_move(PyObject_InlineSupervisor* self, PyObject* args) {
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

static PyObject* InlineSupervisor_player_get_move(PyObject_InlineSupervisor* self, PyObject* args) {
    int lead_role_index = 0;
    if (! ::PyArg_ParseTuple(args, "i", &lead_role_index)) {
        return nullptr;
    }

    int res = self->impl->puctPlayerGetMove(lead_role_index);
    return ::Py_BuildValue("i", res);
    Py_RETURN_NONE;
}


static struct PyMethodDef InlineSupervisor_methods[] = {
    {"start_self_play_test", (PyCFunction) InlineSupervisor_start_self_play_test, METH_VARARGS, "start_self_play_test"},
    {"poll", (PyCFunction) InlineSupervisor_poll, METH_VARARGS, "poll"},

    {"player_start", (PyCFunction) InlineSupervisor_player_start, METH_NOARGS, "player_start"},
    {"player_apply_move", (PyCFunction) InlineSupervisor_player_apply_move, METH_VARARGS, "player_apply_move"},
    {"player_move", (PyCFunction) InlineSupervisor_player_move, METH_VARARGS, "player_move"},
    {"player_get_move", (PyCFunction) InlineSupervisor_player_get_move, METH_VARARGS, "player_get_move"},



    {nullptr, nullptr}            /* Sentinel */
};

static void InlineSupervisor_dealloc(PyObject* ptr);


static PyTypeObject PyType_InlineSupervisor = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "InlineSupervisor",                /*tp_name*/
    sizeof(PyObject_InlineSupervisor), /*tp_size*/
    0,                        /*tp_itemsize*/

    /* methods */
    InlineSupervisor_dealloc,    /*tp_dealloc*/
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
    InlineSupervisor_methods,    /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
};

static PyObject_InlineSupervisor* PyType_InlineSupervisor_new(Supervisor* impl) {
    PyObject_InlineSupervisor* res = PyObject_New(PyObject_InlineSupervisor,
                                                  &PyType_InlineSupervisor);
    res->impl = impl;
    return res;
}


static void InlineSupervisor_dealloc(PyObject* ptr) {
    K273::l_debug("--> InlineSupervisor_dealloc");
    ::PyObject_Del(ptr);
}

///////////////////////////////////////////////////////////////////////////////

static PyObject* gi_InlineSupervisor(PyObject* self, PyObject* args) {
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

    return (PyObject*) PyType_InlineSupervisor_new(supervisor);
}
