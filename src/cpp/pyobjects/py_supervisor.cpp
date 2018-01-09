#pragma once

#include "../supervisor.h"

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

struct PyObject_SupervisorWrapper {
    PyObject_HEAD
    Supervisor* impl;
};

///////////////////////////////////////////////////////////////////////////////

static PyObject* SupervisorWrapper_add(PyObject_SupervisorWrapper* self, PyObject* args) {
    static long data[4] = {0,1,2,3};

    const int ND = 2;
    const int SIZE = 2;
    npy_intp dims[2]{SIZE, SIZE};

    return PyArray_SimpleNewFromData(ND, dims, NPY_INT64, &data);
}

static PyObject* SupervisorWrapper_test_sm(PyObject_SupervisorWrapper* self, PyObject* args) {
    ssize_t ptr = 0;

    if (! ::PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }

    GGPLib::StateMachine* sm = reinterpret_cast<GGPLib::StateMachine*> (ptr);
    return PyString_FromString(sm->getGDL(42));
}

static struct PyMethodDef SupervisorWrapper_methods[] = {
    {"add", (PyCFunction) SupervisorWrapper_add, METH_VARARGS, "blablabla"},
    {"test_sm", (PyCFunction) SupervisorWrapper_test_sm, METH_VARARGS, "blablabla"},
    {nullptr, nullptr}            /* Sentinel */
};

static void SupervisorWrapper_dealloc(PyObject* ptr);


static PyTypeObject PyType_SupervisorWrapper = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "SupervisorWrapper",                /*tp_name*/
    sizeof(PyObject_SupervisorWrapper), /*tp_size*/
    0,                        /*tp_itemsize*/

    /* methods */
    SupervisorWrapper_dealloc,    /*tp_dealloc*/
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
    SupervisorWrapper_methods,    /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
};

static PyObject_SupervisorWrapper* PyType_SupervisorWrapper_new(Supervisor* impl) {
    PyObject_SupervisorWrapper* res = PyObject_New(PyObject_SupervisorWrapper,
                                                   &PyType_SupervisorWrapper);
    res->impl = impl;
    return res;
}


static void SupervisorWrapper_dealloc(PyObject* ptr) {
    K273::l_debug("--> SupervisorWrapper_dealloc");
    ::PyObject_Del(ptr);
}

///////////////////////////////////////////////////////////////////////////////

static PyObject* gi_Supervisor(PyObject* self, PyObject* args) {
    int arg0, arg1;
    if (! ::PyArg_ParseTuple(args, "ii", &arg0, &arg1)) {
        return nullptr;
    }

    // XXXZZZ
    //GdlBasesTransformer* transformer = new GdlBasesTransformer(arg0, arg1);
    return (PyObject *) PyType_SupervisorWrapper_new(nullptr);
}
