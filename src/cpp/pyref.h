#pragma once

// python includes
#include <Python.h>

struct PyCleanupRef {
    PyCleanupRef(PyObject* o) :
        o(o) {
    }

    ~PyCleanupRef() {
        Py_DECREF(o);
    }

    PyObject* o;
};

#define PYCLEANUPREF(x) PyCleanupRef cleanup##x(x);
