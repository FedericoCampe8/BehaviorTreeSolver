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

}  // namespace btsolver
