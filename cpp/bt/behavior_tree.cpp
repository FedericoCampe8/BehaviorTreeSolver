#include "bt/behavior_tree.hpp"

#include <algorithm>  // for std::max
#include <stdexcept>  // for std::invalid_argument

namespace btsolver {

BehaviorTree::BehaviorTree(BehaviorTreeArena::UPtr arena)
: pArena(std::move(arena))
{
  if (pArena == nullptr)
  {
    throw std::invalid_argument("BehaviorTree - empty arena");
  }
}

void BehaviorTree::setEntryNode(uint32_t entryNode)
{
  pEntryNode = entryNode;
}

void BehaviorTree::run()
{
  // Reset the status
  pStatus = NodeStatus::kActive;
  pStopRun = false;

  if (pEntryNode == std::numeric_limits<uint32_t>::max())
  {
    // Entry node not set, return asap
    return;
  }

  auto entryNodePtr = pArena->getNode(pEntryNode);
  if (entryNodePtr == nullptr)
  {
    throw std::runtime_error("BehaviorTree - run: empty entry node");
  }

  // Start ticking
  if (pTotNumTicks > 0)
  {
    for (uint32_t tickCtr{0}; tickCtr < pTotNumTicks; ++tickCtr)
    {
      if (pStopRun)
      {
        break;
      }

      pStatus = entryNodePtr->tick();
    }
  }
  else
  {
    while (pStatus == NodeStatus::kActive)
    {
      if (pStopRun)
      {
        break;
      }

      pStatus = entryNodePtr->tick();
    }
  }
}

}  // namespace btsolver
