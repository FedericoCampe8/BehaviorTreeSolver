#include "bt/behavior_tree.hpp"

#include <algorithm>  // for std::max

namespace btsolver {

void BehaviorTree::setEntryNode(Node::UPtr entryNode)
{
  pEntryNode = std::move(entryNode);
}

void BehaviorTree::run()
{
  // Reset the status
  pStatus = NodeStatus::kActive;
  pStopRun = false;

  if (pEntryNode == nullptr)
  {
    return;
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

      pStatus = pEntryNode->tick();
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

      pStatus = pEntryNode->tick();
    }
  }
}

}  // namespace btsolver
