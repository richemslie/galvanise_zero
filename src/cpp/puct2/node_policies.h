#pragma once

#include "statemachine/statemachine.h"
#include "statemachine/jointmove.h"
#include "statemachine/basestate.h"

#include <k273/util.h>

#include <string>
#include <vector>

namespace GGPZero::PuctV2 {
    typedef float Score;

    constexpr static int lead_role_index_simultaneous = -1;

    // Forwards
    struct PuctNode;

    struct PuctNodeChild {
        PuctNode* to_node;
        bool unselectable;
        uint32_t traversals;

        // this is a cross product of legals.  In a simultaneous game, number of
        // PuctNodeChild != LegalPolicy.
        GGPLib::JointMove move;
    };

    struct LegalPolicyIndex {
        // incr pointer to where we start LegalPolicy
        uint16_t start_incr;

        // the number of legal policies
        uint16_t num_legal_policies;
    };


    struct LegalPolicy {
        uint16_t legal_index;
        float policy_prob;
        float next_prob;

        // dirichlet_noise XXX - should be internal
        // XXXfloat dirichlet_noise;

        // not sure what to do with these XXX
        // XXX Score debug_node_score;
        // XXX Score debug_puct_score;
    };

    typedef std::vector <const PuctNodeChild*> Children;

    inline int round_up_8(int x) {
        if (x % 8 == 0) {
            return x;
        }

        return ((x / 8) + 1) * 8;
    }

    struct PuctNode {
        // actual visits

        const PuctNode* parent;

        // actual visits to this node (differs from traversals, due to transpositions)
        uint32_t visits;

        // visited count, but not been added back in yet (decremented when applying updates)
        uint16_t inflight_visits;

        // needed for transpositions and releasing nodes
        uint16_t ref_count;

        // number of children with unselectables set
        uint16_t unselectable_count;

        uint16_t num_children;
        uint16_t num_children_expanded;

        // whether this node has a finalised scores or not (can also release children if so)
        bool is_finalised;

        // we don't really know which player it really it is for each node, but this is our best guess
        int16_t lead_role_index;

        // the depth of the game
        uint16_t game_depth;

        // internal pointer to scores
        uint16_t final_score_ptr_incr;
        uint16_t basestate_ptr_incr;
        uint16_t children_ptr_incr;

        uint16_t policy_sizes_ptr_incr;
        uint16_t policy_ptr_incr;

        // actual size of this node
        uint16_t allocated_size;

        uint8_t data[0];

    private:
        template <typename T>
        T* getIncr(uint16_t incr) {
            uint8_t* mem = this->data + incr;
            return reinterpret_cast <T*>(mem);
        }

        template <typename T>
        const T* getConstIncr(uint16_t incr) const {
            const uint8_t* mem = this->data + incr;
            return reinterpret_cast <const T*>(mem);
        }

        int nodeChildIncr(const int role_count, const int child_index) const {
            int node_child_bytes = (sizeof(PuctNodeChild) +
                                    role_count * sizeof(GGPLib::JointMove::IndexType));
            node_child_bytes = round_up_8(node_child_bytes);

            return this->children_ptr_incr + node_child_bytes * child_index;
        }

    public:
        Score getCurrentScore(int role_index) const {
            // current score incr is 0
            const Score* scores = this->getConstIncr <Score>(0);
            return *(scores + role_index);
        }

        void setCurrentScore(int role_index, Score score) {
            // current score incr is 0
            Score* scores = this->getIncr <Score>(0);
            *(scores + role_index) = score;
        }

        Score getFinalScore(int role_index) const {
            /* score as per predicted by NN value head, or the terminal scores */
            const Score* scores = this->getConstIncr <Score>(this->final_score_ptr_incr);
            return *(scores + role_index);
        }

        Score setFinalScore(int role_index, Score score) {
            Score* scores = this->getIncr <Score>(this->final_score_ptr_incr);
            return *(scores + role_index);
        }

        GGPLib::BaseState* getBaseState() {
            return this->getIncr <GGPLib::BaseState>(this->basestate_ptr_incr);
        }

        const GGPLib::BaseState* getBaseState() const {
            return this->getConstIncr <GGPLib::BaseState>(this->basestate_ptr_incr);
        }

        uint16_t getPolicySize(const int role_index) const {
            const uint16_t* incrs = this->getConstIncr <uint16_t>(this->policy_sizes_ptr_incr);
            return *(incrs + role_index);
        }

        void setPolicySize(const int role_index, uint16_t size) {
            // XXX need this.  Done at init.
            uint16_t* incrs = this->getIncr <uint16_t>(this->policy_sizes_ptr_incr);
            *(incrs + role_index) = size;
        }

        PlusNodeChild* getLegalPolicy(const int role_index, const int child_index) {
            // XXX hard
            return this->getIncr<PlusNodeChild>(this->nodeChildIncr(role_count, child_index));
        }

        PlusNodeChild* setLegalPolicy(const int role_count, const int child_index) {
            // XXX hard
            return this->getIncr<PlusNodeChild>(this->nodeChildIncr(role_count, child_index));
        }


        PlusNodeChild* getNodeChild(const int role_count, const int child_index) {
            return this->getConstIncr <PlusNodeChild>(this->nodeChildIncr(role_count, child_index));
        }

        const PlusNodeChild* getNodeChild(const int role_count, const int child_index) const {
            return this->getConstIncr <PlusNodeChild>(this->nodeChildIncr(role_count, child_index));
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

        static PuctNode* create(const GGPLib::BaseState* base_state,
                                GGPLib::StateMachineInterface* sm);

        static std::string moveString(const GGPLib::JointMove& move,
                                      GGPLib::StateMachineInterface* sm);

        static void dumpNode(const PuctNode* node, const PuctNodeChild* highlight,
                             const std::string& indent,
                             bool sort_by_next_probability,
                             GGPLib::StateMachineInterface* sm);

        static Children sortedChildren(const PuctNode* node,
                                       int role_count,
                                       bool next_probability=false);
    };

}

