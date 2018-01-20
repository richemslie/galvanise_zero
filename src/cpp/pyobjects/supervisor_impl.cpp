#pragma once

#include "sample.h"

#include <vector>

///////////////////////////////////////////////////////////////////////////////

struct PyObject_Supervisor {
    PyObject_HEAD
    Supervisor* impl;
};

static PyObject* Supervisor_start_self_play(PyObject_Supervisor* self, PyObject* args) {
    int num_workers = 0;
    PyObject* dict;
    if (! PyArg_ParseTuple(args, "iO!", &num_workers, &PyDict_Type, &dict)) {
        return nullptr;
    }

    SelfPlayConfig* config = ::createSelfPlayConfig(dict);
    if (num_workers <= 0) {
        self->impl->createInline(config);

    } else {
        for (int ii=0; ii<num_workers; ii++) {
            self->impl->createWorkers(config);
        }
    }

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

    // populate dict from Sample (prev_states, policy, final_score - done after)
    PyDict_setNewItem(sample_as_dict, "state", stateToList(sample->state));
    PyDict_setNewItem(sample_as_dict, "depth", PyInt_FromLong(sample->depth));
    PyDict_setNewItem(sample_as_dict, "lead_role_index", PyInt_FromLong(sample->lead_role_index));

    PyDict_setNewItem(sample_as_dict, "game_length", PyInt_FromLong(sample->game_length));
    PyDict_setNewItem(sample_as_dict, "match_identifier", PyString_FromString(sample->match_identifier.c_str()));
    PyDict_setNewItem(sample_as_dict, "has_resigned", PyBool_FromLong((int) sample->has_resigned));
    PyDict_setNewItem(sample_as_dict, "resign_false_positive", PyBool_FromLong((int) sample->resign_false_positive));
    PyDict_setNewItem(sample_as_dict, "starting_sample_depth", PyInt_FromLong(sample->starting_sample_depth));


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

static PyObject* Supervisor_add_unique_state(PyObject_Supervisor* self, PyObject* args) {
    ssize_t ptr = 0;
    if (! ::PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }

    GGPLib::BaseState* bs = reinterpret_cast<GGPLib::BaseState*> (ptr);
    self->impl->addUniqueState(bs);
    Py_RETURN_NONE;
}

static PyObject* Supervisor_clear_unique_states(PyObject_Supervisor* self, PyObject* args) {
    self->impl->clearUniqueStates();
    Py_RETURN_NONE;
}

static PyObject* Supervisor_poll(PyObject_Supervisor* self, PyObject* args) {
    return doPoll(self->impl, args);
}

static struct PyMethodDef Supervisor_methods[] = {
    {"start_self_play", (PyCFunction) Supervisor_start_self_play, METH_VARARGS, "start_self_play"},
    {"fetch_samples", (PyCFunction) Supervisor_fetch_samples, METH_NOARGS, "fetch_samples"},

    {"add_unique_state", (PyCFunction) Supervisor_add_unique_state, METH_VARARGS, "add_unique_state"},
    {"clear_unique_states", (PyCFunction) Supervisor_clear_unique_states, METH_NOARGS, "clear_unique_states"},

    {"poll", (PyCFunction) Supervisor_poll, METH_VARARGS, "poll"},

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

    // sm, transformer, batch_size, expected_policy_size, role_1_index
    if (! ::PyArg_ParseTuple(args, "nO!i", &ptr,
                             &(PyType_GdlBasesTransformerWrapper), &obj,
                             &batch_size)) {
        return nullptr;
    }

    PyObject_GdlBasesTransformerWrapper* py_transformer = (PyObject_GdlBasesTransformerWrapper*) obj;

    GGPLib::StateMachine* sm = reinterpret_cast<GGPLib::StateMachine*> (ptr);

    // create the c++ object
    Supervisor* supervisor = new Supervisor(sm, py_transformer->impl, batch_size);
    return (PyObject*) PyType_Supervisor_new(supervisor);
}
