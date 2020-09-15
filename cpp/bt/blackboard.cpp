#include "bt/blackboard.hpp"

namespace {

constexpr uint64_t kDefaultMemoryAllocation{100};

}  // namespace

namespace btsolver {

StateOptimizationQueue::StateOptimizationQueue()
: queuePtr(0)
{
  queue.reserve(kDefaultMemoryAllocation);
}

Blackboard::Blackboard()
: pOptimizationQueue(std::make_shared<StateOptimizationQueue>())
{
  pMostRecentStatesList.reserve(kDefaultMemoryAllocation);
}

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

bool Blackboard::checkState(uint32_t stateId) const noexcept
{
  return pStateMemory.at(stateId);
}

bool Blackboard::checkAndDeactivateState(uint32_t stateId)
{
  const auto val = pStateMemory[stateId];
  pStateMemory[stateId] = false;
  return val;
}

}  // namespace btsolver
