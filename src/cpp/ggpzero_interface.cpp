// python includes
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

///////////////////////////////////////////////////////////////////////////////
// global variables

PyObject* ggpzero_interface_error;

///////////////////////////////////////////////////////////////////////////////

// these are python objects... we include cpp files since it is hard to do any
// other way (please if you know a way, let me know)

#include "pyobjects/common.cpp"
#include "pyobjects/gdltransformer_impl.cpp"
#include "pyobjects/player_impl.cpp"
#include "pyobjects/player_impl2.cpp"
#include "pyobjects/supervisor_impl.cpp"


static PyObject* gi_buf_to_tuple_reverse_bytes(PyObject* self, PyObject* args) {
    PyObject* buf_object = nullptr;
    if (! ::PyArg_ParseTuple(args, "O", &buf_object)) {
        return nullptr;
    }

    ASSERT(PyString_Check(buf_object));
    const char* ptbuf = PyString_AsString(buf_object);

    const int buf_size = PyString_GET_SIZE(buf_object);
    PyObject* tup = PyTuple_New(buf_size * 8);

    for (int ii=0; ii<buf_size; ii++, ptbuf++) {
        const char c = *ptbuf;
        const int incr = ii * 8;
        for (int jj=0; jj<8; jj++) {
            PyObject* obj = (c & (1 << (7 - jj))) ? Py_True : Py_False;
            Py_INCREF(obj);
            const int index = jj + incr;
            PyTuple_SET_ITEM(tup, index, obj);
        }
    }

    return tup;
}

///////////////////////////////////////////////////////////////////////////////

PyMethodDef gi_functions[] = {
    {"buf_to_tuple_reverse_bytes", gi_buf_to_tuple_reverse_bytes, METH_VARARGS, "buf_to_tuple_reverse_bytes"},
    {"GdlBasesTransformer", gi_GdlBasesTransformer, METH_VARARGS, "GdlBasesTransformer"},
    {"Player", gi_Player, METH_VARARGS, "Player"},
    {"Player2", gi_Player2, METH_VARARGS, "Player2"},
    {"Supervisor", gi_Supervisor, METH_VARARGS, "Supervisor"},

    {nullptr, nullptr, 0, nullptr}
};

///////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

    PyMODINIT_FUNC initggpzero_interface() {
        Py_TYPE(&PyType_GdlBasesTransformerWrapper) = &PyType_Type;

        if (::PyType_Ready(&PyType_GdlBasesTransformerWrapper) < 0) {
            return;
        }

        Py_TYPE(&PyType_Player) = &PyType_Type;

        if (::PyType_Ready(&PyType_Player) < 0) {
            return;
        }

        Py_TYPE(&PyType_Player2) = &PyType_Type;

        if (::PyType_Ready(&PyType_Player2) < 0) {
            return;
        }

        Py_TYPE(&PyType_Supervisor) = &PyType_Type;

        if (::PyType_Ready(&PyType_Supervisor) < 0) {
            return;
        }

        PyObject* m = ::Py_InitModule("ggpzero_interface", gi_functions);
        if (m == nullptr) {
            return;
        }

        char error_name[] = "ggpzero_interface.error";
        ggpzero_interface_error = ::PyErr_NewException(error_name, nullptr, nullptr);
        Py_INCREF(ggpzero_interface_error);
        ::PyModule_AddObject(m, "GGPZeroInterfaceError", ggpzero_interface_error);

        import_array();
    }
}
