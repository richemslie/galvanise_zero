
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

#ifdef __cplusplus
extern "C" {
#endif

enum greenlet_flags
{
    GREENLET_STARTED = 0x1,
    GREENLET_DEAD = 0x2
};

typedef struct _greenlet_st greenlet_t;
typedef void *(*greenlet_start_func_t)(void *);
typedef void (*greenlet_inject_func_t)(void *);

struct _greenlet_st
{
    greenlet_t *gr_parent;
    void *gr_stack;
    long gr_stacksize;
    int gr_flags;
    greenlet_start_func_t gr_start;
    void *gr_arg;
    void *gr_instance;
    greenlet_inject_func_t gr_inject;
    void *gr_frame[8];
};

greenlet_t *greenlet_new(greenlet_start_func_t start_func,
                         greenlet_t *parent, long stacksize);
void greenlet_destroy(greenlet_t *greenlet);

void *greenlet_switch_to(greenlet_t *greenlet, void *arg);
void greenlet_inject(greenlet_t *greenlet, greenlet_inject_func_t inject_func);
void greenlet_reset(greenlet_t *greenlet);

greenlet_t *greenlet_root();
greenlet_t *greenlet_current();
greenlet_t *greenlet_parent(greenlet_t *greenlet);

int greenlet_isstarted(greenlet_t *greenlet);
int greenlet_isdead(greenlet_t *greenlet);

#ifdef __cplusplus
}
#endif


class greenlet_exit: public std::exception {
};

struct _greenlet_data
{
#ifdef HAVE_CXX11
    std::exception_ptr exception;
#else
    greenlet_exit *exception;
#endif
};

class greenlet {
public:
    greenlet(greenlet_start_func_t start_func=0L,
             greenlet *parent=0L, int stacksize=0);
    greenlet(greenlet_t *greenlet);
    virtual ~greenlet();

    static greenlet *root();
    static greenlet *current();
    greenlet *parent();

    void *switch_to(void *arg=0L);
    void inject(greenlet_inject_func_t inject_func);
    void reset();

    bool isstarted();
    bool isdead();

protected:
    virtual void *run(void *arg);

private:
    static void *_run(void *arg);
    static void _inject_exception(void *arg);

    greenlet_t *_greenlet;
    greenlet_start_func_t _start_func;
    _greenlet_data *_data;
};
