#pragma once

#include "../dummysupervisor.h"

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

using namespace GGPZero;

///////////////////////////////////////////////////////////////////////////////

struct PyObject_DummySupervisorWrapper {
    PyObject_HEAD
    SupervisorDummy* impl;
};

///////////////////////////////////////////////////////////////////////////////

static PyObject* DummySupervisorWrapper_test(PyObject_DummySupervisorWrapper* self, PyObject* args) {
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

    int res = self->impl->test(policies, final_scores, sz);

    if (res) {
        npy_intp dims[1]{res};
        return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, self->impl->getBuf());
    } else {
        Py_RETURN_NONE;
    }
}

static struct PyMethodDef DummySupervisorWrapper_methods[] = {
    {"test", (PyCFunction) DummySupervisorWrapper_test, METH_VARARGS, "test"},
    {nullptr, nullptr}            /* Sentinel */
};

static void DummySupervisorWrapper_dealloc(PyObject* ptr);


static PyTypeObject PyType_DummySupervisorWrapper = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "DummySupervisorWrapper",                /*tp_name*/
    sizeof(PyObject_DummySupervisorWrapper), /*tp_size*/
    0,                        /*tp_itemsize*/

    /* methods */
    DummySupervisorWrapper_dealloc,    /*tp_dealloc*/
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
    DummySupervisorWrapper_methods,    /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
};

static PyObject_DummySupervisorWrapper* PyType_DummySupervisorWrapper_new(SupervisorDummy* impl) {
    PyObject_DummySupervisorWrapper* res = PyObject_New(PyObject_DummySupervisorWrapper,
                                                        &PyType_DummySupervisorWrapper);
    res->impl = impl;
    return res;
}


static void DummySupervisorWrapper_dealloc(PyObject* ptr) {
    K273::l_debug("--> DummySupervisorWrapper_dealloc");
    ::PyObject_Del(ptr);
}

///////////////////////////////////////////////////////////////////////////////

static PyObject* gi_SupervisorDummy(PyObject* self, PyObject* args) {
    ssize_t ptr = 0;
    PyObject* obj = 0;
    int batch_size = 0;
    int expected_policy_size = 0;
    int role_1_index = 0;

    // sm, transformer, batch_size
    if (! ::PyArg_ParseTuple(args, "nO!iii", &ptr,
                             &(PyType_GdlBasesTransformerWrapper), &obj,
                             &batch_size, &expected_policy_size, &role_1_index)) {
        return nullptr;
    }

    PyObject_GdlBasesTransformerWrapper* py_transformer = (PyObject_GdlBasesTransformerWrapper*) obj;

    GGPLib::StateMachine* sm = reinterpret_cast<GGPLib::StateMachine*> (ptr);

    // create the thing
    SupervisorDummy* dummy = new SupervisorDummy(sm, py_transformer->impl, batch_size,
                                                 expected_policy_size, role_1_index);
    return (PyObject*) PyType_DummySupervisorWrapper_new(dummy);
}
