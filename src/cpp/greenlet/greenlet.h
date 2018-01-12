
/*
 * This file is part of cgreenlet. CGreenlet is free software available
 * under the terms of the MIT license. Consult the file LICENSE that was
 * shipped together with this source file for the exact licensing terms.
 *
 * Copyright (c) 2012 by the cgreenlet authors. See the file AUTHORS for a
 * full list.
 */

#pragma once

#include "greenlet-int.h"

#include <exception>
#include <functional>

enum greenlet_flags
{
    GREENLET_STARTED = 0x1,
    GREENLET_DEAD = 0x2
};

typedef struct _greenlet_st greenlet_t;
typedef void *(*greenlet_start_func_t)(void *);

struct _greenlet_st
{
    greenlet_t *gr_parent;
    void *gr_stack;
    long gr_stacksize;
    int gr_flags;
    greenlet_start_func_t gr_start;
    void *gr_arg;
    void *gr_instance;
    void *gr_frame[8];
};

greenlet_t *greenlet_new(greenlet_start_func_t start_func,
                         greenlet_t *parent, long stacksize);
void greenlet_destroy(greenlet_t *greenlet);

void *greenlet_switch_to(greenlet_t *greenlet, void *arg=nullptr);
void greenlet_reset(greenlet_t *greenlet);

greenlet_t *greenlet_root();
greenlet_t *greenlet_current();
greenlet_t *greenlet_parent(greenlet_t *greenlet);

int greenlet_isstarted(greenlet_t *greenlet);
int greenlet_isdead(greenlet_t *greenlet);


// simple wrapper machinery to use closures from c++.  The signature of the closure is:
// f(void) -> void
// Since no more is needed.

namespace _greenlet_hidden {
    template <typename Callable> struct WrapClosure {
        typedef WrapClosure <Callable> Self;

        Callable function;
        greenlet_t* bounce;

        WrapClosure(const Callable& function) :
            function(function),
            bounce(greenlet_current()) {
        }

        static void* call(void* data) {
            Self* wrapper = reinterpret_cast<Self*>(data);
            greenlet_switch_to(wrapper->bounce);
            wrapper->function();
            return nullptr;
        }
    };

    typedef WrapClosure <std::function<void(void)>> WrapSimpleLambda;
}


template <typename Callable>
greenlet_t* createGreenlet(const Callable& f, greenlet_t* parent=nullptr) {
    greenlet_t* g = greenlet_new(_greenlet_hidden::WrapSimpleLambda::call, parent, 0);
    _greenlet_hidden::WrapSimpleLambda* arg = new _greenlet_hidden::WrapSimpleLambda(f);
    greenlet_switch_to(g, arg);
    return g;
}
