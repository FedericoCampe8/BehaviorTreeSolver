#include "bt/blackboard.hpp"

namespace btsolver {

NodeStatus Blackboard::getNodeStatus(uint32_t nodeId) noexcept
{
  auto iter = pNodeStatusMap.find(nodeId);
  if (iter == pNodeStatusMap.end())
  {
    pNodeStatusMap[nodeId] = NodeStatus::kPending;
    return NodeStatus::kPending;
  }
  return iter->second;
}

void Blackboard::addState(uint32_t stateId, bool isActive)
{
  pStateMemory[stateId] = isActive;
}  // addState

bool Blackboard::checkAndDeactivateState(uint32_t stateId)
{
  const auto val = pStateMemory[stateId];
  pStateMemory[stateId] = false;
  return val;
}

}  // namespace btsolver
