#pragma once

#include "gdltransformer.h"

// k273 includes
#include <k273/rng.h>
#include <k273/logging.h>

// ggplib imports
#include <statemachine/basestate.h>
#include <statemachine/jointmove.h>
#include <statemachine/propagate.h>
#include <statemachine/legalstate.h>
#include <statemachine/statemachine.h>

#include <vector>

using namespace GGPZero;

///////////////////////////////////////////////////////////////////////////////

struct PyObject_GdlBasesTransformerWrapper {
    PyObject_HEAD
    GdlBasesTransformer* impl;
};

static PyObject* GdlBasesTransformerWrapper_addBaseInfo(PyObject_GdlBasesTransformerWrapper* self, PyObject* args) {
    int arg0, arg1;
    if (! ::PyArg_ParseTuple(args, "ii", &arg0, &arg1)) {
        return nullptr;
    }

    self->impl->addBaseInfo(arg0, arg1);

    return Py_None;
}

static PyObject* GdlBasesTransformerWrapper_addControlState(PyObject_GdlBasesTransformerWrapper* self, PyObject* args) {
    int arg0;
    if (! ::PyArg_ParseTuple(args, "i", &arg0)) {
        return nullptr;
    }

    self->impl->addControlState(arg0);

    return Py_None;
}

static PyObject* GdlBasesTransformerWrapper_test(PyObject_GdlBasesTransformerWrapper* self, PyObject* args) {

    const int MOVES = 4096;
    static float* array_buf = nullptr;

    if (array_buf == nullptr) {
        array_buf = (float*) malloc(sizeof(float) * self->impl->totalSize() * MOVES);
    }

    ssize_t ptr = 0;
    int prev_states = 0;
    if (! ::PyArg_ParseTuple(args, "ni", &ptr, &prev_states)) {
        return nullptr;
    }

    GGPLib::StateMachine* sm = reinterpret_cast<GGPLib::StateMachine*> (ptr);

    GGPLib::JointMove* joint_move = sm->getJointMove();
    GGPLib::BaseState* other = sm->newBaseState();
    const GGPLib::BaseState* bs = sm->getInitialState();
    other->assign(bs);
    sm->updateBases(bs);

    std::vector <GGPLib::BaseState*> dummy;

    float* pt_array_buf = array_buf;

    // four random moves
    K273::xoroshiro128plus32 random;

    int total_depth = 0;
    for (int jj=0; jj<2; jj++) {
        sm->reset();

        for (int kk=0; kk<MOVES; kk++) {

            if (sm->isTerminal()) {
                break;
            }

            // populate joint move
            for (int ii=0; ii<sm->getRoleCount(); ii++) {
                const GGPLib::LegalState* ls = sm->getLegalState(ii);
                int x = random.getWithMax(ls->getCount());
                int choice = ls->getLegal(x);
                joint_move->set(ii, choice);
            }

            sm->nextState(joint_move, other);
            sm->updateBases(other);

            if (prev_states) {
                std::vector <GGPLib::BaseState*> prevs;
                for (int ii=0; ii<prev_states; ii++) {
                    prevs.push_back(other);
                }

                self->impl->toChannels(other, prevs, pt_array_buf);

            } else {
                self->impl->toChannels(other, dummy, pt_array_buf);
            }

            pt_array_buf += self->impl->totalSize();
            total_depth++;
        }
    }


    const int ND = 1;
    npy_intp dims[1]{self->impl->totalSize() * total_depth};

    return PyArray_SimpleNewFromData(ND, dims, NPY_FLOAT, array_buf);
}

static struct PyMethodDef GdlBasesTransformerWrapper_methods[] = {
    {"add_base_info", (PyCFunction) GdlBasesTransformerWrapper_addBaseInfo, METH_VARARGS, "addBaseInfo"},
    {"add_control_state", (PyCFunction) GdlBasesTransformerWrapper_addControlState, METH_VARARGS, "addControlState"},
    {"test", (PyCFunction) GdlBasesTransformerWrapper_test, METH_VARARGS, "test"},
    {nullptr, nullptr}            /* Sentinel */
};

static void GdlBasesTransformerWrapper_dealloc(PyObject* ptr);


static PyTypeObject PyType_GdlBasesTransformerWrapper = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "GdlBasesTransformerWrapper",                /*tp_name*/
    sizeof(PyObject_GdlBasesTransformerWrapper), /*tp_size*/
    0,                        /*tp_itemsize*/

    /* methods */
    GdlBasesTransformerWrapper_dealloc,    /*tp_dealloc*/
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
    GdlBasesTransformerWrapper_methods,    /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
};

static PyObject_GdlBasesTransformerWrapper* PyType_GdlBasesTransformerWrapper_new(GdlBasesTransformer* impl) {
    PyObject_GdlBasesTransformerWrapper* res = PyObject_New(PyObject_GdlBasesTransformerWrapper,
                                                            &PyType_GdlBasesTransformerWrapper);
    res->impl = impl;
    return res;
}


static void GdlBasesTransformerWrapper_dealloc(PyObject* ptr) {
    K273::l_debug("--> GdlBasesTransformerWrapper_dealloc");
    ::PyObject_Del(ptr);
}

///////////////////////////////////////////////////////////////////////////////

static PyObject* gi_GdlBasesTransformer(PyObject* self, PyObject* args) {
    int channel_size, channels_per_state, num_prev_states;
    PyObject* expected_policy_sizes;

    if (! ::PyArg_ParseTuple(args, "iiiO!",
                             &channel_size,
                             &channels_per_state,
                             &num_prev_states,
                             &PyList_Type, &expected_policy_sizes)) {
        return nullptr;
    }

    auto asInt = [expected_policy_sizes] (int index) {
        PyObject* borrowed = PyList_GET_ITEM(expected_policy_sizes, index);
        return PyInt_AsLong(borrowed);
    };

    std::vector <int> policy_sizes;
    for (int ii=0; ii<PyList_Size(expected_policy_sizes); ii++) {
        policy_sizes.push_back(asInt(ii));
    }

    GdlBasesTransformer* transformer = new GdlBasesTransformer(channel_size,
                                                               channels_per_state,
                                                               num_prev_states,
                                                               policy_sizes);

    return (PyObject *) PyType_GdlBasesTransformerWrapper_new(transformer);
}
