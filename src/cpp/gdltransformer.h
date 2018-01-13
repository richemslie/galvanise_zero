/* Note: there is no shape sizes here.  That is because we just return a continous numpy array to
   python and let the python code reshape it how it likes. */

#pragma once

#include "puct/node.h"

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
        GdlBasesTransformer(int channel_size, int control_states_start, int expected_policy_size, int role_1_index) :
            channel_size(channel_size),
            control_states_start(control_states_start),
            expected_policy_size(expected_policy_size),
            role_1_index(role_1_index) {
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
                        float* buf) const;

        int getPolicySize() const {
            return this->expected_policy_size;
        }

        int getRole1Index() const {
            return this->role_1_index;
        }

    private:
        const int channel_size;
        const int control_states_start;
        const int expected_policy_size;
        const int role_1_index;
        std::vector <int> control_states;
        std::vector <BaseInfo> base_infos;
    };

}
