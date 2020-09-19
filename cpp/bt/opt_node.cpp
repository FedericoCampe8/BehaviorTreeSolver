#include "bt/opt_node.hpp"

#include <algorithm>  // for std::min, std::max
#include <cassert>
#include <stdexcept>  // for std::invalid_argument

namespace btsolver {
namespace optimization {

OptimizationStateCondition::OptimizationStateCondition(const std::string& name,
                                                       BehaviorTreeArena* arena)
: Node(name, NodeType::ConditionState, arena)
{
  // Register the run callback
  registerRunCallback([=]() {
    return this->runNode();
  });

  // Register the cleanup callback
  registerCleanupCallback([=]() {
    return this->cleanupNode();
  });
}

void OptimizationStateCondition::pairWithOptimizationState(Node* state)
{
  assert(state != nullptr);
  pPairedState.push_back(state->cast<OptimizationState>());

  // Register this condition node with the paired state
  pPairedState.back()->pairStateConditionNode(this);
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

NodeStatus OptimizationStateCondition::runNode()
{
  return pIsActive ? NodeStatus::kSuccess : NodeStatus::kFail;
}

void OptimizationStateCondition::cleanupNode()
{
  // Reset the active flag
  pIsActive = false;
}

OptimizationState::OptimizationState(const std::string& name, BehaviorTreeArena* arena)
: Node(name, NodeType::State, arena),
  pDefaultDPState(std::make_shared<DPState>())
{
  pDPState = pDefaultDPState;

  // Register the run callback
  registerRunCallback([=]() {
    return this->runNode();
  });
}

NodeStatus OptimizationState::runNode()
{
  // This node does two main things:
  // 1) activates the paired state condition (if any); and
  // 2) updates lower and upper bounds on the cost function
  // Where, to set the bounds on the cost function:
  // 2.1 - set the bounds of this node incoming edges summing the values from
  //       this variable's domain and the parent state condition node (if any)
  // 2.2 - Copy over the bounds to this node
  // 2.2 - Set the bounds on the paired state condition node (if any)
  //
  // Note:
  // This state can have multiple incoming edges (which, in turn can be parallel or not)
  // since this state can be shared among different selectors of the same child.
  // In other words, this state can hold onto a DP state that ends up to be the same
  // when activated by different parent conditions.
  // For example, in the AllDifferent constraint, the DP state {1, 2}, coming from the path
  // x_1 = {1}, x_2 = {2}, is the same as the DP state {2, 1}, coming from the path
  // x_1 = {2}, x_2 = {1}.
  // Therefore, a different edge should be considered at each tick
  const auto& allEdges = getAllIncomingEdges();
  assert(pParentConditionsList.empty() || pParentConditionsList.size() == allEdges.size());

  if (pCurrentTickedEdge > allEdges.size())
  {
    // Reset the counter
    pCurrentTickedEdge = 0;
  }

  // Notice that the parent state condition list is either empty (first child,
  // no parent states activating a first child node), or equal to the number
  // of incoming edges (one condition state for each edge leading to this node)
  auto parentStateCondition = pParentConditionsList.empty() ?
          nullptr :pParentConditionsList.at(pCurrentTickedEdge);
  auto inEdge = allEdges.at(pCurrentTickedEdge++);
  assert(inEdge != nullptr);

  if (inEdge->isParallelEdge())
  {
    // The incoming edge is a parallel edge.
    // Take the lower and upper bounds on the cost
    // according to the lower/upper bounds domain values
    // on the parallel edge
    auto cost = inEdge->getCostBounds();
    pLowerBoundCost = cost.first;
    pUpperBoundCost = cost.second;
  }
  else
  {
    // The incoming edge has a single value which
    // represents the value that leads to this state
    // with a correspondent cost
    pLowerBoundCost = inEdge->getCostValue();
    pUpperBoundCost = pLowerBoundCost;
  }

  // Initialize the total lower/upper bound costs representing
  // the total cost of the solution when this state is taken
  pTotLowerBoundCost = pLowerBoundCost;
  pTotUpperBoundCost = pUpperBoundCost;

  // Sum the cost so far, coming from the parent state condition node (if any)
  if (parentStateCondition != nullptr)
  {
    pTotLowerBoundCost += parentStateCondition->getGlbLowerBoundOnCost();
    pTotUpperBoundCost += parentStateCondition->getGlbUpperBoundOnCost();
  }

  // As last step on this node, activate the paired state condition,
  // i.e., the condition node on the right-side brother for the next variable (if any)
  if (pPairedStateCondition != nullptr)
  {
    // Activate the paired state condition node
    pPairedStateCondition->activate();
    pPairedStateCondition->setGlbLowerBoundOnCost(pTotLowerBoundCost);
    pPairedStateCondition->setGlbUpperBoundOnCost(pTotUpperBoundCost);
  }

  // Return FAIL
  return NodeStatus::kFail;
}

RunnerOptimizer::RunnerOptimizer(const std::string& name, BehaviorTreeArena* arena)
: Behavior(name, NodeType::OptimizationRunner, arena)
{
  // Register the run callback
  registerRunCallback([=]() {
    return this->runOptimizer();
  });
}

NodeStatus RunnerOptimizer::runOptimizer()
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

    /*
    std::cout << "RunnerOptimizer - Optimality GAP:\n";
    for (auto node : pOptimizationQueue->queue)
    {
      auto stateNode = reinterpret_cast<OptimizationState*>(node);
      std::cout << "(" << stateNode->getGlbLowerBoundOnCost() << ", " <<
              stateNode->getGlbUpperBoundOnCost() << ")\n";
    }
    */
  }

  // All children run
  return NodeStatus::kSuccess;
}


}  // namespace optimization
}  // btsolver
