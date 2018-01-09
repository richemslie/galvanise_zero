#include "bases.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/exception.h>

#include <string>
#include <vector>

using namespace GGPZero;

void GdlBasesTransformer::toChannels(const GGPLib::BaseState* bs,
                                     const std::vector <GGPLib::BaseState*>& prev_states,
                                     float* buf) {

    // XXX ZZZ only supports 'channel first'.  If we really want to have 'channel last',
    // transformation will need to be via python code on numpy arrays.  Keeping this code lean and
    // fast.

    // prev_states -> list of states
    ASSERT_MSG(prev_states.size() == 0, "unhandled for now");

    // zero out channels for board states
    for (int ii=0; ii<this->control_states_start; ii++) {
        *(buf + ii) = 0.0f;
    }

    // simply add to appropriate channel
    for (size_t ii=0; ii<this->base_infos.size(); ii++) {
        const BaseInfo& binfo = this->base_infos[ii];
        if (binfo.hasChannel() && bs->get(ii)) {
            binfo.set(buf);
        }
    }

    // set the control states
    int channel_idx = this->control_states_start;
    for (auto control_state_idx : this->control_states) {
        float value = bs->get(control_state_idx) ? 1.0f : 0.0f;
        for (int ii=0; ii<this->channel_size; ii++, channel_idx++) {
            *(buf + channel_idx) = value;
        }
    }
}


