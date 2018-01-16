#pragma once

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <mutex>
#include <vector>

namespace GGPZero {
    class UniqueStates {
    public:
        UniqueStates(const GGPLib::StateMachineInterface* sm) :
            sm(sm->dupe()) {
        }

        ~UniqueStates() {
            delete this->sm;
        }

    public:
        void add(const GGPLib::BaseState* bs) {
            std::lock_guard <std::mutex> lk(this->mut);

            GGPLib::BaseState::HashSet::const_iterator it = this->unique_states.find(bs);
            if (it != this->unique_states.end()) {
                return;
            }

            // create a new basestate...
            GGPLib::BaseState* new_bs = this->sm->newBaseState();
            new_bs->assign(bs);
            this->states_allocated.push_back(new_bs);

            this->unique_states.insert(new_bs);
        }

        bool isUnique(GGPLib::BaseState* bs) {
            std::lock_guard <std::mutex> lk(this->mut);
            GGPLib::BaseState::HashSet::const_iterator it = this->unique_states.find(bs);
            return it == this->unique_states.end();
        }

        void clear() {
            std::lock_guard <std::mutex> lk(this->mut);

            for (GGPLib::BaseState* bs : this->states_allocated) {
                ::free(bs);
            }

            this->unique_states.clear();
            this->states_allocated.clear();
        }

    private:
        const GGPLib::StateMachineInterface* sm;
        GGPLib::BaseState::HashSet unique_states;
        std::mutex mut;
        std::vector <GGPLib::BaseState*> states_allocated;
    };
}
