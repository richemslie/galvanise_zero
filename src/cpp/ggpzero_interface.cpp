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

///////////////////////////////////////////////////////////////////////////////

PyMethodDef gi_functions[] = {
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
