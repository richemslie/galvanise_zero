/*
 * This file is part of cgreenlet. CGreenlet is free software available
 * under the terms of the MIT license. Consult the file LICENSE that was
 * shipped together with this source file for the exact licensing terms.
 *
 * Copyright (c) 2012 by the cgreenlet authors. See the file AUTHORS for a
 * full list.
 */


#include "greenlet.h"
#include "greenlet-int.h"

#include <stdlib.h>

#include <new>
#include <exception>

/* On systems with __thread support, getting/setting the current thread is
 * optimized using inline functions. */

#ifdef TLS_USE___THREAD

static __thread greenlet_t _root_greenlet =
{ NULL, NULL, 0, GREENLET_STARTED };
static __thread greenlet_t *_current_greenlet = NULL;

inline greenlet_t *_greenlet_get_root()
{
    return &_root_greenlet;
}

inline greenlet_t *_greenlet_get_current()
{
    return _current_greenlet;
}

inline void _greenlet_set_current(greenlet_t *current)
{
    _current_greenlet = current;
}

#endif  /* TLS_USE___THREAD */


greenlet_t *greenlet_new(greenlet_start_func_t start_func,
                         greenlet_t *parent, long stacksize)
{
    greenlet_t *greenlet;

    greenlet = (greenlet_t *) calloc(1, sizeof(greenlet_t));
    if (greenlet == NULL)
        return NULL;
    greenlet->gr_parent = parent ? parent : _greenlet_get_root();
    if (greenlet->gr_parent == NULL)
        return NULL;
    greenlet->gr_start = start_func;
    greenlet->gr_stacksize = stacksize;
    greenlet->gr_stack = _greenlet_alloc_stack(&greenlet->gr_stacksize);
    if (greenlet->gr_stack == NULL)
        return NULL;
    return greenlet;
}

void greenlet_destroy(greenlet_t *greenlet)
{
    _greenlet_dealloc_stack(greenlet->gr_stack, greenlet->gr_stacksize);
}

static void _greenlet_start(void *arg)
{
    greenlet_t *greenlet = (greenlet_t *) arg;
    void *ret;

    _greenlet_set_current(greenlet);
    greenlet->gr_flags |= GREENLET_STARTED;
    if (greenlet->gr_inject)
        greenlet->gr_inject(greenlet->gr_arg);
    greenlet->gr_inject = NULL;
    ret = greenlet->gr_start(greenlet->gr_arg);
    greenlet->gr_flags |= GREENLET_DEAD;

    while (greenlet->gr_flags & GREENLET_DEAD)
        greenlet = greenlet->gr_parent;

    greenlet->gr_arg = ret;
    _greenlet_set_current(greenlet);
    _greenlet_switchcontext(&greenlet->gr_frame, greenlet->gr_inject, ret);
}

void *greenlet_switch_to(greenlet_t *greenlet, void *arg)
{
    greenlet_t *current = greenlet_current();

    if (_greenlet_savecontext(&current->gr_frame))
    {
        current->gr_inject = NULL;
        return current->gr_arg;
    }

    if (!(greenlet->gr_flags & GREENLET_STARTED))
    {
        greenlet->gr_arg = arg;
        _greenlet_newstack((char *) greenlet->gr_stack + greenlet->gr_stacksize,
                           _greenlet_start, greenlet);
    }

    while (greenlet->gr_flags & GREENLET_DEAD)
        greenlet = greenlet->gr_parent;

    greenlet->gr_arg = arg;
    _greenlet_set_current(greenlet);
    _greenlet_switchcontext(&greenlet->gr_frame, greenlet->gr_inject, arg);
}

void greenlet_inject(greenlet_t *greenlet, greenlet_inject_func_t inject_func)
{
    greenlet->gr_inject = inject_func;
}

void greenlet_reset(greenlet_t *greenlet)
{
    greenlet->gr_flags = 0;
}

greenlet_t *greenlet_root()
{
    return _greenlet_get_root();
}

greenlet_t *greenlet_current()
{
    greenlet_t *greenlet = _greenlet_get_current();
    if (greenlet == NULL)
        greenlet = _greenlet_get_root();
    return greenlet;
}

greenlet_t *greenlet_parent(greenlet_t *greenlet)
{
    return greenlet->gr_parent;
}

int greenlet_isstarted(greenlet_t *greenlet)
{
    return (greenlet->gr_flags & GREENLET_STARTED) > 0;
}

int greenlet_isdead(greenlet_t *greenlet)
{
    return (greenlet->gr_flags & GREENLET_DEAD) > 0;
}

greenlet::greenlet(greenlet_t *greenlet)
{
    _greenlet = greenlet;
    _greenlet->gr_instance = this;
    _start_func = greenlet->gr_start;
    greenlet->gr_start = _run;
    _data = new _greenlet_data;
}

greenlet::greenlet(greenlet_start_func_t start_func, greenlet *parent,
                   int stacksize)
{
    greenlet_t *c_parent = parent ? parent->_greenlet->gr_parent : 0L;

    _greenlet = greenlet_new(_run, c_parent, stacksize);
    if (_greenlet == 0L)
        throw std::bad_alloc();
    _greenlet->gr_instance = this;
    _start_func = start_func;
    _data = new _greenlet_data;
}

greenlet::~greenlet()
{
    greenlet_destroy(_greenlet);
    delete _data;
}

greenlet *greenlet::root()
{
    greenlet_t *c_root = greenlet_root();
    greenlet *root = (greenlet *) c_root->gr_instance;
    if (root == 0L)
        root = new greenlet(c_root);
    return root;
}

greenlet *greenlet::current()
{
    greenlet_t *c_current = greenlet_current();
    greenlet *current = (greenlet *) c_current->gr_instance;
    if (current == 0L)
        current = new greenlet(c_current);
    return current;
}

greenlet *greenlet::parent()
{
    greenlet_t *c_parent = _greenlet->gr_parent;
    greenlet *parent = (greenlet *) c_parent->gr_instance;
    if (parent == 0L)
        parent = new greenlet(c_parent);
    return parent;
}

void *greenlet::switch_to(void *arg)
{
    return greenlet_switch_to(_greenlet, arg);
}

void greenlet::inject(greenlet_inject_func_t inject_func)
{
    _greenlet->gr_inject = inject_func;
}

void greenlet::reset()
{
    _greenlet->gr_flags = 0;
}

bool greenlet::isstarted()
{
    return (_greenlet->gr_flags & GREENLET_STARTED) > 0;
}

bool greenlet::isdead()
{
    return (_greenlet->gr_flags & GREENLET_DEAD) > 0;
}

void *greenlet::run(void *arg)
{
    if (_start_func == 0L)
        return 0L;
    return _start_func(arg);
}

void *greenlet::_run(void *arg)
{
    greenlet *greenlet = current();

    try {
        return greenlet->run(arg);
    } catch (...) {
        greenlet = greenlet->parent();
        while (greenlet->isdead())
            greenlet = greenlet->parent();
        greenlet->_greenlet->gr_inject = _inject_exception;
#ifdef HAVE_CXX11
        greenlet->_data->exception = std::current_exception();
#else
        greenlet->_data->exception = new greenlet_exit();
#endif
        return 0L;
    }
}

void greenlet::_inject_exception(void *arg)
{
#ifdef HAVE_CXX11
    std::rethrow_exception(current()->_data->exception);
#else
    throw *current()->_data->exception;
#endif
}
