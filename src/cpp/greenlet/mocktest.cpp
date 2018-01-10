#include "mocktest.h"
#include "greenlet.h"

#include <k273/util.h>
#include <k273/logging.h>

const static int PINGPONGS = 1000000;

class greenlet1: public greenlet {
public:
    int count = 0;
    void *run(void *arg) {
        greenlet* other = (greenlet*) arg;
        for (int ii=0; ii<PINGPONGS; ii++) {
            other = (greenlet*) other->switch_to(this);
            count++;
        }

        return nullptr;
    }
};

class greenlet2: public greenlet {
public:
    int count = 0;
    void *run(void *arg) {
        greenlet* other = (greenlet*) arg;
        for (int ii=0; ii<PINGPONGS; ii++) {
            other = (greenlet*) other->switch_to(this);
            count++;
        }

        return nullptr;
    }
};


class X {
public:
    X() {
        this->gr1 = new greenlet1;
        this->gr2 = new greenlet2;
    }

    ~X() {
        delete this->gr1;
        delete this->gr2;
    }

    void go() {
        this->gr1->switch_to(this->gr2);
    }


    greenlet1* gr1;
    greenlet2* gr2;
};


void mainLoop() {
    double enter_time = K273::get_time();
    while (true) {
        if (K273::get_time() > enter_time + 5) {
            break;
        }

        X x;

        double s = K273::get_time();
        x.go();
        double taken_msecs = (K273::get_time() - s) * 1000;

        // let's just say it is fast:
        K273::l_verbose("here %d %d,  pingpong %d switches %.3f",
                        x.gr1->count, x.gr2->count, PINGPONGS, taken_msecs);
    }
}

namespace GGPZero {

    void test_cgreenlet() {
        std::vector <std::thread*> threads;
        const int NUM_THREADS = 8;
        for (int ii=0; ii<NUM_THREADS; ii++) {
            threads.push_back(new std::thread(&mainLoop));
        }

        for (auto t : threads) {
            t->join();
        }
    }
}
