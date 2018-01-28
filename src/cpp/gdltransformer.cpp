#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/exception.h>

#include <string>
#include <vector>

using namespace GGPZero;


void GdlBasesTransformer::setForState(float* local_buf, const GGPLib::BaseState* bs) const {
    for (size_t ii=0; ii<this->base_infos.size(); ii++) {
        const BaseInfo& binfo = this->base_infos[ii];
        if (binfo.hasChannel() && bs->get(ii)) {
            binfo.set(local_buf);
        }
    }
}

void GdlBasesTransformer::toChannels(const GGPLib::BaseState* the_base_state,
                                     const std::vector <GGPLib::BaseState*>& prev_states,
                                     float* buf) const {

    // NOTE: only supports 'channel first'.  If we really want to have 'channel last',
    // transformation will need to be via python code on numpy arrays.  Keeping this code lean and
    // fast.

    // zero out channels for board states
    const int zero_floats = (this->channels_per_state *
                             (this->num_prev_states + 1) *
                             this->channel_size);

    for (int ii=0; ii<zero_floats; ii++) {
        *(buf + ii) = 0.0f;
    }

    this->setForState(buf, the_base_state);

    int count = 1;
    for (const GGPLib::BaseState* b : prev_states) {
        float* local_buf = buf + (this->channels_per_state * this->channel_size * count);
        this->setForState(local_buf, b);
    }

    // set the control states
    int channel_idx = this->controlStatesStart();
    for (auto control_state_idx : this->control_states) {
        float value = the_base_state->get(control_state_idx) ? 1.0f : 0.0f;
        for (int ii=0; ii<this->channel_size; ii++, channel_idx++) {
            *(buf + channel_idx) = value;
        }
    }
}


