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
        GdlBasesTransformer(int channel_size,
                            int channels_per_state,
                            int num_prev_states,
                            std::vector <int>& expected_policy_sizes) :
            channel_size(channel_size),
            channels_per_state(channels_per_state),
            num_prev_states(num_prev_states),
            expected_policy_sizes(expected_policy_sizes) {
        }

    public:
        // builder methods
        void addBaseInfo(bool has_channel, int index) {
            this->base_infos.emplace_back(has_channel, index);
        }

        void addControlState(int index) {
            this->control_states.push_back(index);
        }

    private:
        void setForState(float* local_buf, const GGPLib::BaseState* bs) const;

        int controlStatesStart() const {
            return this->channel_size * (this->channels_per_state * (this->num_prev_states + 1));
        }

    public:
        // client side methods
        int totalSize() const {
            return this->channel_size * (this->channels_per_state * (this->num_prev_states + 1) +
                                         this->control_states.size());
        }

        void toChannels(const GGPLib::BaseState* bs,
                        const std::vector <const GGPLib::BaseState*>& prev_states,
                        float* buf) const;

        int getNumberPrevStates() const {
            return this->num_prev_states;
        }

        // whether policy info should be on this, i dunno.  guess following python's lead.  XXX
        int getNumberPolicies() const {
            return this->expected_policy_sizes.size();
        }

        int getPolicySize(int i) const {
            return this->expected_policy_sizes[i];
        }

    private:
        const int channel_size;
        const int channels_per_state;
        const int num_prev_states;
        std::vector <int> control_states;
        std::vector <BaseInfo> base_infos;

        std::vector <int> expected_policy_sizes;
    };

}
