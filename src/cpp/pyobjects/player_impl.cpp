#pragma once

struct PyObject_Player {
    PyObject_HEAD
    GGPZero::Player* impl;
};

static PyObject* Player_reset(PyObject_Player* self, PyObject* args) {
    int game_depth = 0;
    if (! ::PyArg_ParseTuple(args, "i", &game_depth)) {
        return nullptr;
    }

    self->impl->puctPlayerReset(game_depth);
    Py_RETURN_NONE;
}

static PyObject* Player_apply_move(PyObject_Player* self, PyObject* args) {
    ssize_t ptr = 0;
    if (! ::PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }

    GGPLib::JointMove* move = reinterpret_cast<GGPLib::JointMove*> (ptr);
    self->impl->puctApplyMove(move);

    Py_RETURN_NONE;
}

static PyObject* Player_move(PyObject_Player* self, PyObject* args) {
    ssize_t ptr = 0;
    int evaluations = 0;
    double end_time = 0.0;
    if (! ::PyArg_ParseTuple(args, "nid", &ptr, &evaluations, &end_time)) {
        return nullptr;
    }

    GGPLib::BaseState* basestate = reinterpret_cast<GGPLib::BaseState*> (ptr);
    self->impl->puctPlayerMove(basestate, evaluations, end_time);
    Py_RETURN_NONE;
}

static PyObject* Player_get_move(PyObject_Player* self, PyObject* args) {
    int lead_role_index = 0;
    if (! ::PyArg_ParseTuple(args, "i", &lead_role_index)) {
        return nullptr;
    }

    int a, c;
    float b;
    std::tie(a, b, c) = self->impl->puctPlayerGetMove(lead_role_index);
    return ::Py_BuildValue("ifi", a, b, c);
}

static PyObject* Player_balance_moves(PyObject_Player* self, PyObject* args) {
    int max_count = 0;
    if (! ::PyArg_ParseTuple(args, "i", &max_count)) {
        return nullptr;
    }

    self->impl->balanceNode(max_count);

    Py_RETURN_NONE;
}

#include <stdio.h>
static PyObject* Player_tree_debug(PyObject_Player* self, PyObject* args) {
    int max_count = 0;
    if (! ::PyArg_ParseTuple(args, "i", &max_count)) {
        return nullptr;
    }

    std::vector <PuctNodeDebug> debug_list = self->impl->treeDebugInfo(max_count);

    PyObject* result = PyTuple_New(debug_list.size());

    for (int ii=0; ii<debug_list.size(); ii++) {
        const PuctNodeDebug& debug = debug_list[ii];

        PyObject* variation = PyTuple_New(debug.variation.size());
        for (int jj=0; jj<debug.variation.size(); jj++) {
            PyObject* variation_desc = ::Py_BuildValue("ii",
                                                       debug.variation[jj].lead_role_index,
                                                       debug.variation[jj].move_index);
            PyTuple_SetItem(variation, jj, variation_desc);
        }

        PyObject* el = ::Py_BuildValue("fiiO",
                                       debug.score,
                                       debug.lead_role_index,
                                       debug.move_index,
                                       variation);

        PyTuple_SetItem(result, ii, el);
    }

    return result;
}

static PyObject* Player_updateConfig(PyObject_Player* self, PyObject* args) {
    double think_time = 0.0f;
    int converge_relaxed = 0.0f;
    int verbose = 0;
    if (! ::PyArg_ParseTuple(args, "dii", &think_time, &converge_relaxed, &verbose)) {
        return nullptr;
    }

    self->impl->updateConfig(think_time, converge_relaxed, verbose);
    Py_RETURN_NONE;
}

static PyObject* Player_poll(PyObject_Player* self, PyObject* args) {
    return doPoll(self->impl, args);
}

static struct PyMethodDef Player_methods[] = {
    {"player_reset", (PyCFunction) Player_reset, METH_VARARGS, "player_reset"},
    {"player_update_config", (PyCFunction) Player_updateConfig, METH_VARARGS, "player_update_config"},
    {"player_apply_move", (PyCFunction) Player_apply_move, METH_VARARGS, "player_apply_move"},
    {"player_move", (PyCFunction) Player_move, METH_VARARGS, "player_move"},
    {"player_get_move", (PyCFunction) Player_get_move, METH_VARARGS, "player_get_move"},

    {"player_balance_moves", (PyCFunction) Player_balance_moves, METH_VARARGS, "player_balance_moves"},
    {"player_tree_debug", (PyCFunction) Player_tree_debug, METH_VARARGS, "player_get_move"},

    {"poll", (PyCFunction) Player_poll, METH_VARARGS, "poll"},

    {nullptr, nullptr}            /* Sentinel */
};

static void Player_dealloc(PyObject* ptr);

static PyTypeObject PyType_Player = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "Player",                /*tp_name*/
    sizeof(PyObject_Player), /*tp_size*/
    0,                        /*tp_itemsize*/

    /* methods */
    Player_dealloc,    /*tp_dealloc*/
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
    Player_methods,    /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
};

static PyObject_Player* PyType_Player_new(GGPZero::Player* impl) {
    PyObject_Player* res = PyObject_New(PyObject_Player,
                                        &PyType_Player);
    res->impl = impl;
    return res;
}


static void Player_dealloc(PyObject* ptr) {
    K273::l_debug("--> Player_dealloc");
    ::PyObject_Del(ptr);
}

///////////////////////////////////////////////////////////////////////////////

static PyObject* gi_Player(PyObject* self, PyObject* args) {
    ssize_t ptr = 0;
    PyObject_GdlBasesTransformerWrapper* py_transformer = nullptr;
    PyObject* dict = nullptr;

    // sm, transformer, batch_size, expected_policy_size, role_1_index
    if (! ::PyArg_ParseTuple(args, "nO!O!", &ptr,
                             &PyType_GdlBasesTransformerWrapper, &py_transformer,
                             &PyDict_Type, &dict)) {
        return nullptr;
    }

    GGPLib::StateMachine* sm = reinterpret_cast<GGPLib::StateMachine*> (ptr);
    GGPZero::PuctConfig* conf = createPuctConfig(dict);

    // create the c++ object
    GGPZero::Player* player = new GGPZero::Player(sm, py_transformer->impl, conf);
    return (PyObject*) PyType_Player_new(player);
}
