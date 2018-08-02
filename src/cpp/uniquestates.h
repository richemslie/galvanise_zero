#pragma once

#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <mutex>
#include <vector>

namespace GGPZero {
    class UniqueStates {
    public:
        UniqueStates(const GGPLib::StateMachineInterface* sm,
                     const GGPZero::GdlBasesTransformer* transformer,
                     int max_num_dupes=1) :
            sm(sm->dupe()),
            max_num_dupes(max_num_dupes) {

            this->lookup = GGPLib::BaseState::makeMaskedMap <int>(transformer->createHashMask(this->sm->newBaseState()));
        }

        ~UniqueStates() {
            delete this->sm;
        }

    public:
        void add(const GGPLib::BaseState* bs) {
            std::lock_guard <std::mutex> lk(this->mut);

            auto it = this->lookup->find(bs);
            if (it != this->lookup->end()) {
                if (it->second < max_num_dupes) {
                    it->second += 1;
                }

                return;
            }

            // create a new basestate...
            GGPLib::BaseState* new_bs = this->sm->newBaseState();
            new_bs->assign(bs);
            this->states_allocated.push_back(new_bs);

            this->lookup->emplace(new_bs, 1);
        }

        bool isUnique(GGPLib::BaseState* bs, int depth) {
            std::lock_guard <std::mutex> lk(this->mut);
            const auto it = this->lookup->find(bs);
            if (it != this->lookup->end()) {
                int allowed_dupes = std::max(2, (this->max_num_dupes - 5 * depth));
                if (it->second >= allowed_dupes) {
                    return false;
                }
            }

            return true;
        }

        void clear() {
            std::lock_guard <std::mutex> lk(this->mut);

            for (GGPLib::BaseState* bs : this->states_allocated) {
                ::free(bs);
            }

            this->lookup->clear();
            this->states_allocated.clear();
        }

    private:
        const GGPLib::StateMachineInterface* sm;
        const int max_num_dupes;

        std::mutex mut;
        GGPLib::BaseState::HashMapMasked <int>* lookup;
        std::vector <GGPLib::BaseState*> states_allocated;
    };
}
