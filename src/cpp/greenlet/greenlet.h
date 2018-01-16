
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

#include <k273/logging.h>
#include <k273/exception.h>

#include <type_traits>
#include <typeinfo>
#include <cxxabi.h>
#include <exception>
#include <functional>
#include <string>

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

template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
            abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
            std::free
            );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

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

            try {
                wrapper->function();
            } catch (const K273::Exception& exc) {
                std::string wrapper_name = type_name<decltype(wrapper)>();
                K273::l_critical("In %s, Exception: %s",
                                 wrapper_name.c_str(),
                                 exc.getMessage().c_str());
                throw;

            } catch (...) {
                std::string wrapper_name = type_name<decltype(wrapper)>();
                K273::l_critical("In %s, Unknown exception caught.",
                                 wrapper_name.c_str());
                throw;
            }

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
