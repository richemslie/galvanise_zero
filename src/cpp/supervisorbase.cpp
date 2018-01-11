#include "supervisorbase.h"
#include "puct/node.h"

using namespace GGPZero;

void SupervisorBase::dumpNode(const PuctNode* node,
                              const PuctNodeChild* highlight,
                              const std::string& indent) {
    PuctNode::dumpNode(node, highlight, indent, this->sm);
}
