/* Note: there is no shape sizes here.  That is because we just return a continous numpy array to
   python and let the python code reshape it how it likes. */

#pragma once

#include "puctnode.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <string>
#include <vector>


namespace GGPZero {

    class BaseInfo {
    public:
        BaseInfo(bool has_channel, int index) :
            has_channel(has_channel),
            index(index) {
        }
    public:
        void set(float* buf) const {
            *(buf + this->index) = 1.0f;
        }

        bool hasChannel() const {
            return this->has_channel;
        }

    private:
        bool has_channel;

        // index into buffer ... channels[b_info.channel][b_info.y_idx][b_info.x_idx]
        int index;
    };

    class GdlBasesTransformer {
    public:
        GdlBasesTransformer(int channel_size, int control_states_start) :
            channel_size(channel_size),
            control_states_start(control_states_start) {
        }

    public:
        void addBaseInfo(bool has_channel, int index) {
            this->base_infos.emplace_back(has_channel, index);
        }

        void addControlState(int index) {
            this->control_states.push_back(index);
        }

        int totalSize() const {
            return this->control_states_start + this->channel_size * this->control_states.size();
        }

        void toChannels(const GGPLib::BaseState* bs,
                        const std::vector <GGPLib::BaseState*>& prev_states,
                        float* buf);

    private:
        int channel_size;
        int control_states_start;
        std::vector <int> control_states;
        std::vector <BaseInfo> base_infos;
    };

}
