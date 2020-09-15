#include "bt/opt_node.hpp"

#include <algorithm>  // for std::min, std::max
#include <stdexcept>  // for std::invalid_argument

namespace btsolver {
namespace optimization {

OptimizationStateCondition::OptimizationStateCondition(const std::string& name,
                                                       BehaviorTreeArena* arena,
                                                       Blackboard* blackboard)
: Node(name, arena, blackboard)
{
  // Register the run callback
  registerRunCallback([=](Blackboard* bb) {
    return this->runOptimizationStateConditionNode(bb);
  });

  // Register the cleanup callback
  registerCleanupCallback([=](Blackboard* bb) {
    return this->cleanupNode(bb);
  });
}

void OptimizationStateCondition::pairWithOptimizationState(uint32_t state)
{
  pPairedState.push_back(
          reinterpret_cast<optimization::OptimizationState*>(getArena()->getNode(state)));

  // Register this condition node with the paired state
  pPairedState.back()->pairStateConditionNode(this->getUniqueId());
}

void OptimizationStateCondition::setGlbLowerBoundOnCost(double lb) noexcept
{
  // This node can be activated by multiple state nodes.
  // Hence, the total cost can come from multiple states
  pTotLowerBoundCost = std::min<double>(lb, pTotLowerBoundCost);
}

/// Sets the upper bound on the solution cost
void OptimizationStateCondition::setGlbUpperBoundOnCost(double ub) noexcept
{
  // This node can be activated by multiple state nodes.
  // Hence, the total cost can come from multiple states
  pTotUpperBoundCost = std::max<double>(ub, pTotUpperBoundCost);
}

NodeStatus OptimizationStateCondition::runOptimizationStateConditionNode(Blackboard* blackboard)
{
  if (pIsActive)
  {
    return NodeStatus::kSuccess;
  }
  return NodeStatus::kFail;
}

void OptimizationStateCondition::cleanupNode(Blackboard* blackboard)
{
  // Reset the active flag
  pIsActive = false;
}

OptimizationState::OptimizationState(const std::string& name, BehaviorTreeArena* arena,
                                     Blackboard* blackboard)
: Node(name, arena, blackboard),
  pDPState(std::make_shared<DPState>())
{
  // Register the run callback
  registerRunCallback([=](Blackboard* bb) {
    return this->runOptimizationStateNode(bb);
  });
}

void OptimizationState::pairStateConditionNode(uint32_t stateCondition) noexcept
{
  pPairedStateCondition = reinterpret_cast<optimization::OptimizationStateCondition*>(
          getArena()->getNode(stateCondition));
}

NodeStatus OptimizationState::runOptimizationStateNode(Blackboard* blackboard)
{
  // This node does two main things:
  // 1) activates the paired state condition (if any); and
  // 2) updates lower and upper bounds on the cost function
  // Where, to set the bounds on the cost function:
  // 2.1 - set the bounds of this node incoming edges summing the values from
  //       this variable's domain and the parent state condition node (if any)
  // 2.2 - Copy over the bounds to this node
  // 2.2 - Set the bounds on the paired state condition node (if any)
  auto inEdgeId = getIncomingEdge();
  if (inEdgeId != std::numeric_limits<uint32_t>::max())
  {
    auto edge = getArena()->getEdge(inEdgeId);

    // Check if the edge is a parallel one
    if (edge->isParallelEdge())
    {
      auto cost = edge->getCostBounds();
      pLowerBoundCost = cost.first;
      pUpperBoundCost = cost.second;
    }
    else
    {
      pLowerBoundCost = edge->getCostValue();
      pUpperBoundCost = pLowerBoundCost;
    }
    pTotLowerBoundCost = pLowerBoundCost;
    pTotUpperBoundCost = pUpperBoundCost;
  }

  if (pParentStateConditionNode != std::numeric_limits<uint32_t>::max())
  {
    auto parentOptState = reinterpret_cast<OptimizationStateCondition*>(
            getArena()->getNode(pParentStateConditionNode));
    pTotLowerBoundCost += parentOptState->getGlbLowerBoundOnCost();
    pTotUpperBoundCost += parentOptState->getGlbUpperBoundOnCost();
  }

  if (pPairedStateCondition)
  {
    // Activate the paired state condition node
    pPairedStateCondition->activate();
    pPairedStateCondition->setGlbLowerBoundOnCost(pTotLowerBoundCost);
    pPairedStateCondition->setGlbUpperBoundOnCost(pTotUpperBoundCost);
  }

  // Push this node into the queue of state nodes processed
  // by the runner optimizer node
  // TODO Check if this is really needed
  blackboard->getOptimizationQueueMutable()->queue.push_back(this);

  // Return FAIL
  return NodeStatus::kFail;
}

RunnerOptimizer::RunnerOptimizer(const std::string& name, BehaviorTreeArena* arena, Blackboard* blackboard)
: Behavior(name, arena, blackboard),
  pOptimizationQueue(blackboard->getOptimizationQueue())
{
  if(pOptimizationQueue == nullptr)
  {
    throw std::invalid_argument("RunnerOptimizer - empty optimization queue");
  }

  // Register the run callback
  registerRunCallback([=](Blackboard* bb) {
    return this->runOptimizer(bb);
  });
}

NodeStatus RunnerOptimizer::runOptimizer(Blackboard* blackboard)
{
  // Reset the status of all children before running
  resetChildrenStatus();

  // Run one child at a time, from left to right
  for (int idx{0}; idx < static_cast<int>(getChildren().size()); ++idx)
  {
    auto child = getChildren()[idx];

    auto result = tickChild(child);
    if (result == NodeStatus::kActive || result == NodeStatus::kPending)
    {
      // The current child is still active, return asap
      return NodeStatus::kActive;
    }

    std::cout << "RunnerOptimizer - Optimality GAP:\n";
    for (auto node : pOptimizationQueue->queue)
    {
      auto stateNode = reinterpret_cast<OptimizationState*>(node);
      std::cout << "(" << stateNode->getGlbLowerBoundOnCost() << ", " <<
              stateNode->getGlbUpperBoundOnCost() << ")\n";
    }

    if (idx < static_cast<int>(getChildren().size()) - 1)
    {
      // Store the last states for backtracking assignment
      pOptimizationQueue->queue.clear();
    }
  }

  // All children run
  return NodeStatus::kSuccess;
}


}  // namespace optimization
}  // btsolver
