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
    // ref counts all stolen
    PyObject* state_as_list = PyList_New(bs->size);
    for (int ii=0; ii<bs->size; ii++) {
        PyList_SetItem(state_as_list, ii, PyInt_FromLong(bs->get(ii)));
    }

    return state_as_list;
}

template <typename V, typename F>
PyObject* createListAndPopulate(const V& vec, const F& fn) {
    int count = vec.size();
    PyObject* l = PyList_New(count);
    for (int ii=0; ii<count; ii++) {
        // fn creates a new python object, set item on list steal the ref count.
        PyList_SetItem(l, ii, fn(vec[ii]));
    }

    return l;
}

PyObject* policyElementToTuple(const std::pair<int, float>& e) {
    // ref counts all stolen :)
    PyObject* tup = PyTuple_New(2);
    PyTuple_SetItem(tup, 0, PyInt_FromLong(e.first));
    PyTuple_SetItem(tup, 1, PyFloat_FromDouble((double) e.second));
    return tup;
}

void PyDict_setNewItem(PyObject* d, const char* name, PyObject* o) {
    // steals ref
    ASSERT(PyDict_SetItemString(d, name, o) == 0);
    Py_DECREF(o);
}


PyObject* toPolicies(const Sample::Policy& policy) {
    return createListAndPopulate(policy, policyElementToTuple);
}

PyObject* sampleToDict(Sample* sample) {
    // awesome - trying to steal as much ref counts as possible, while keeping the code as clean
    // as possible!

    PyObject* sample_as_dict = PyDict_New();

    PyDict_setNewItem(sample_as_dict, "state",
                      stateToList(sample->state));

    PyDict_setNewItem(sample_as_dict, "prev_states",
                      createListAndPopulate(sample->prev_states, stateToList));

    PyDict_setNewItem(sample_as_dict, "policies",
                      createListAndPopulate(sample->policies, toPolicies));

    PyDict_setNewItem(sample_as_dict, "final_score",
                      createListAndPopulate(sample->final_score, PyFloat_FromDouble));

    PyDict_setNewItem(sample_as_dict, "depth",
                      PyInt_FromLong(sample->depth));

    PyDict_setNewItem(sample_as_dict, "game_length",
                      PyInt_FromLong(sample->game_length));

    PyDict_setNewItem(sample_as_dict, "match_identifier",
                      PyString_FromString(sample->match_identifier.c_str()));

    PyDict_setNewItem(sample_as_dict, "has_resigned",
                      PyBool_FromLong((int) sample->has_resigned));

    PyDict_setNewItem(sample_as_dict, "resign_false_positive",
                      PyBool_FromLong((int) sample->resign_false_positive));

    PyDict_setNewItem(sample_as_dict, "starting_sample_depth",
                      PyInt_FromLong(sample->starting_sample_depth));

    PyDict_setNewItem(sample_as_dict, "resultant_puct_score",
                      createListAndPopulate(sample->resultant_puct_score, PyFloat_FromDouble));

    PyDict_setNewItem(sample_as_dict, "resultant_puct_visits",
                      PyInt_FromLong(sample->resultant_puct_visits));

    return sample_as_dict;
}

static PyObject* Supervisor_fetch_samples(PyObject_Supervisor* self, PyObject* args) {
    std::vector <Sample*> samples = self->impl->getSamples();

    if (!samples.empty()) {
        return createListAndPopulate(samples, sampleToDict);
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
    const char* indentifier = nullptr;

    // sm, transformer, batch_size, expected_policy_size, role_1_index
    if (! ::PyArg_ParseTuple(args, "nO!is", &ptr,
                             &(PyType_GdlBasesTransformerWrapper), &obj,
                             &batch_size, &indentifier)) {
        return nullptr;
    }

    PyObject_GdlBasesTransformerWrapper* py_transformer = (PyObject_GdlBasesTransformerWrapper*) obj;

    GGPLib::StateMachine* sm = reinterpret_cast<GGPLib::StateMachine*> (ptr);

    // create the c++ object
    Supervisor* supervisor = new Supervisor(sm, py_transformer->impl,
                                            batch_size, indentifier);

    return (PyObject*) PyType_Supervisor_new(supervisor);
}
