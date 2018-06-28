#include "gdltransformer.h"

#include <statemachine/basestate.h>
#include <statemachine/statemachine.h>

#include <k273/exception.h>

#include <string>
#include <vector>

using namespace GGPZero;


void GdlBasesTransformer::setForState(float* local_buf, const GGPLib::BaseState* bs) const {
    for (const auto& b : this->board_space) {
        if (b.check(bs)) {
            b.set(local_buf);
        }
    }
}

void GdlBasesTransformer::toChannels(const GGPLib::BaseState* the_base_state,
                                     const std::vector <const GGPLib::BaseState*>& prev_states,
                                     float* buf) const {

    // NOTE: only supports 'channel first'.  If we really want to have 'channel last',
    // transformation will need to be via python code on numpy arrays.  Keeping this code lean and
    // fast.

    // zero out channels for board states
    for (int ii=0; ii<this->totalSize(); ii++) {
        *(buf + ii) = 0.0f;
    }

    // the_base_state
    this->setForState(buf, the_base_state);

    // prev_states
    int count = 1;
    for (const GGPLib::BaseState* b : prev_states) {
        float* local_buf = buf + (this->channels_per_state * this->channel_size * count);
        this->setForState(local_buf, b);
    }

    // set the control states
    float* control_buf_start = buf + this->controlStatesStart();
    for (const auto& c : this->control_space) {
        if (c.check(the_base_state)) {
            c.floodFill(control_buf_start, this->channel_size);
        }
    }
}

GGPLib::BaseState::ArrayType* GdlBasesTransformer::createHashMask(GGPLib::BaseState* bs) const {
    // first we set bs true for everything we are interested in
    for (int ii=0; ii<bs->size; ii++) {
        bs->set(ii, this->interested_set.find(ii) != this->interested_set.end());
    }

    GGPLib::BaseState::ArrayType* buf = (GGPLib::BaseState::ArrayType*) malloc(bs->byte_count);
    memcpy(buf, bs->data, bs->byte_count);
    return buf;
}
