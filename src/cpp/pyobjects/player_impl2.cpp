#pragma once

struct PyObject_Player2 {
    PyObject_HEAD
    GGPZero::PuctV2::Player* impl;
};

static PyObject* Player2_reset(PyObject_Player2* self, PyObject* args) {
    int game_depth = 0;
    if (! ::PyArg_ParseTuple(args, "i", &game_depth)) {
        return nullptr;
    }

    self->impl->puctPlayerReset(game_depth);
    Py_RETURN_NONE;
}

static PyObject* Player2_apply_move(PyObject_Player2* self, PyObject* args) {
    ssize_t ptr = 0;
    if (! ::PyArg_ParseTuple(args, "n", &ptr)) {
        return nullptr;
    }

    GGPLib::JointMove* move = reinterpret_cast<GGPLib::JointMove*> (ptr);
    self->impl->puctApplyMove(move);

    Py_RETURN_NONE;
}

static PyObject* Player2_move(PyObject_Player2* self, PyObject* args) {
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

static PyObject* Player2_get_move(PyObject_Player2* self, PyObject* args) {
    int lead_role_index = 0;
    if (! ::PyArg_ParseTuple(args, "i", &lead_role_index)) {
        return nullptr;
    }

    int a, c;
    float b;
    std::tie(a, b, c) = self->impl->puctPlayerGetMove(lead_role_index);
    return ::Py_BuildValue("ifi", a, b, c);
}

static PyObject* Player2_updateConfig(PyObject_Player2* self, PyObject* args) {
    double think_time = 0.0f;
    int converge_relaxed = 0.0f;
    int converge_non_relaxed = 0.0f;
    int verbose = 0;
    if (! ::PyArg_ParseTuple(args, "diii", &think_time, &converge_relaxed,
                             &converge_non_relaxed, &verbose)) {
        return nullptr;
    }

    self->impl->updateConfig(think_time, converge_relaxed, converge_non_relaxed, verbose);
    Py_RETURN_NONE;
}

static PyObject* Player2_poll(PyObject_Player2* self, PyObject* args) {
    return doPoll(self->impl, args);
}

static struct PyMethodDef Player2_methods[] = {
    {"player_reset", (PyCFunction) Player2_reset, METH_VARARGS, "player_reset"},
    {"player_update_config", (PyCFunction) Player2_updateConfig, METH_VARARGS, "player_update_config"},
    {"player_apply_move", (PyCFunction) Player2_apply_move, METH_VARARGS, "player_apply_move"},
    {"player_move", (PyCFunction) Player2_move, METH_VARARGS, "player_move"},
    {"player_get_move", (PyCFunction) Player2_get_move, METH_VARARGS, "player_get_move"},

    {"poll", (PyCFunction) Player2_poll, METH_VARARGS, "poll"},

    {nullptr, nullptr}            /* Sentinel */
};

static void Player2_dealloc(PyObject* ptr);

static PyTypeObject PyType_Player2 = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "Player2",                /*tp_name*/
    sizeof(PyObject_Player2), /*tp_size*/
    0,                        /*tp_itemsize*/

    /* methods */
    Player2_dealloc,    /*tp_dealloc*/
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
    Player2_methods,    /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
};

static PyObject_Player2* PyType_Player2_new(GGPZero::PuctV2::Player* impl) {
    PyObject_Player2* res = PyObject_New(PyObject_Player2,
                                         &PyType_Player2);
    res->impl = impl;
    return res;
}


static void Player2_dealloc(PyObject* ptr) {
    K273::l_debug("--> Player2_dealloc");
    ::PyObject_Del(ptr);
}

///////////////////////////////////////////////////////////////////////////////

static PyObject* gi_Player2(PyObject* self, PyObject* args) {
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
    GGPZero::PuctV2::PuctConfig* conf = createPuctConfigV2(dict);

    // create the c++ object
    GGPZero::PuctV2::Player* player = new GGPZero::PuctV2::Player(sm, py_transformer->impl, conf);
    return (PyObject*) PyType_Player2_new(player);
}
