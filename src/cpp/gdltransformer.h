/* Note: there is no shape sizes here.  That is because we just return a continous numpy array to
   python and let the python code reshape it how it likes. */

#pragma once

#include "puct/node.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <string>
#include <vector>

namespace GGPZero {

    class BaseToBoardSpace {
    public:
        BaseToBoardSpace(int base_indx, int buf_incr) :
            base_indx(base_indx),
            buf_incr(buf_incr) {
        }

    public:
        bool check(const GGPLib::BaseState* bs) const {
            return bs->get(this->base_indx);
        }

        void set(float* buf) const {
            *(buf + this->buf_incr) = 1.0f;
        }

    private:
        int base_indx;

        // increment into buffer ... channels[b_info.channel][b_info.y_idx][b_info.x_idx]
        int buf_incr;
    };

    class BaseToChannelSpace {
    public:
        BaseToChannelSpace(int base_indx, int channel_id, float value) :
            base_indx(base_indx),
            channel_id(channel_id),
            value(value) {
        }

    public:
        bool check(const GGPLib::BaseState* bs) const {
            return bs->get(this->base_indx);
        }

        void floodFill(float* buf, int channel_size) const {
            buf += channel_size * this->channel_id;
            for (int ii=0; ii<channel_size; ii++, buf++) {
                *buf = this->value;
            }
        }

    private:
        // which base it is
        int base_indx;

        // which channel it is (relative to end of states)
        int channel_id;

        // the value to set the entire channel (flood fill)
        float value;
    };

    class GdlBasesTransformer {
    public:
        GdlBasesTransformer(int channel_size,
                            int channels_per_state,
                            int num_control_channels,
                            int num_prev_states,
                            std::vector <int>& expected_policy_sizes) :
            channel_size(channel_size),
            channels_per_state(channels_per_state),
            num_control_channels(num_control_channels),
            num_prev_states(num_prev_states),
            expected_policy_sizes(expected_policy_sizes) {
        }

    public:
        // builder methods
        void addBoardBase(int base_indx, int buf_incr) {
            this->board_space.emplace_back(base_indx, buf_incr);
        }

        void addControlBase(int base_indx, int channel_id, float value) {
            this->control_space.emplace_back(base_indx, channel_id, value);
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
                                         this->num_control_channels);
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
        const int num_control_channels;
        const int num_prev_states;
        std::vector <BaseToBoardSpace> board_space;
        std::vector <BaseToChannelSpace> control_space;

        std::vector <int> expected_policy_sizes;
    };

}
