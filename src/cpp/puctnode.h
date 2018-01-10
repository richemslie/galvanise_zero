#pragma once

#include "statemachine/statemachine.h"
#include "statemachine/jointmove.h"
#include "statemachine/basestate.h"

#include <k273/util.h>

namespace GGPZero {
    typedef double Score;

    const int LEAD_ROLE_INDEX_SIMULTANEOUS = -1;

    // Forwards
    struct PuctNode;

    struct PuctNodeChild {
        PuctNode* to_node;
        float policy_prob;
        float dirichlet_noise;

        Score debug_node_score;
        Score debug_puct_score;
        GGPLib::JointMove move;
    };

    inline int round_up_8(int x) {
        if (x % 8 == 0) {
            return x;
        }

        return ((x / 8) + 1) * 8;
    }

    struct PuctNode {
        // actual visits
        int visits;

        uint16_t num_children;
        uint16_t num_children_expanded;

        // whether this node has a finalised scores or not (can also release children if so)
        bool is_finalised;

        // we don't really know which player it really it is for each node, but this is our best guess
        int16_t lead_role_index;

        // internal pointer to scores
        uint16_t final_score_ptr_incr;
        uint16_t basestate_ptr_incr;
        uint16_t children_ptr_incr;

        // actual size of this node
        int allocated_size;

        uint8_t data[0];


        Score getCurrentScore(int role_index) const {
            const uint8_t* mem = this->data;
            const Score* scores = reinterpret_cast<const Score*> (mem);
            return *(scores + role_index);
        }

        void setCurrentScore(int role_index, Score score) {
            uint8_t* mem = this->data;

            Score* scores = reinterpret_cast<Score*> (mem);
            *(scores + role_index) = score;
        }

        Score getFinalScore(int role_index) const {
            /* score as per predicted by NN value head, or the terminal scores */
            const uint8_t* mem = this->data;
            mem += this->final_score_ptr_incr;
            const Score* scores = reinterpret_cast<const Score*> (mem);
            return *(scores + role_index);
        }

        Score setFinalScore(int role_index, Score score) {
            /* score as per predicted by NN value head, or the terminal scores */
            uint8_t* mem = this->data;
            mem += this->final_score_ptr_incr;

            const Score* scores = reinterpret_cast<const Score*> (mem);
            return *(scores + role_index);
        }

        GGPLib::BaseState* getBaseState() {
            uint8_t* mem = this->data;
            mem += this->basestate_ptr_incr;
            return reinterpret_cast<GGPLib::BaseState*> (mem);
        }

        const GGPLib::BaseState* getBaseState() const {
            const uint8_t* mem = this->data;
            mem += this->basestate_ptr_incr;
            return reinterpret_cast<const GGPLib::BaseState*> (mem);
        }

        PuctNodeChild* getNodeChild(const int role_count, const int child_index) {
            int node_child_bytes = (sizeof(PuctNodeChild) +
                                    role_count * sizeof(GGPLib::JointMove::IndexType));
            node_child_bytes = round_up_8(node_child_bytes);

            uint8_t* mem = this->data;
            mem += this->children_ptr_incr;
            mem += node_child_bytes * child_index;
            return reinterpret_cast<PuctNodeChild*> (mem);
        }

        const PuctNodeChild* getNodeChild(const int role_count, const int child_index) const {
            int node_child_bytes = (sizeof(PuctNodeChild) +
                                    role_count * sizeof(GGPLib::JointMove::IndexType));
            node_child_bytes = round_up_8(node_child_bytes);

            const uint8_t* mem = this->data;
            mem += this->children_ptr_incr;
            mem += node_child_bytes * child_index;
            return reinterpret_cast<const PuctNodeChild*> (mem);
        }

        bool isTerminal() const {
            return this->num_children == 0;
        }

        static PuctNode* create(int role_count,
                                const GGPLib::BaseState* base_state,
                                GGPLib::StateMachineInterface* sm);

        static void dumpNode(const PuctNode* node, const PuctNodeChild* highlight,
                             const std::string& indent,
                             GGPLib::StateMachineInterface* sm);
    };

}

