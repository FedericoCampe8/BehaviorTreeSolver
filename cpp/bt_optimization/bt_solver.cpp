#include "bt_optimization/bt_solver.hpp"

#include <limits>     // for std::numeric_limits
#include <stdexcept>  // for std::runtime_error
#include <string>

#include "bt/branch.hpp"
#include "bt/opt_node.hpp"
#include "bt/edge.hpp"
#include "bt/node_status.hpp"
#include "cp/bitmap_domain.hpp"
#include "cp/domain.hpp"
#include "cp/variable.hpp"

namespace btsolver {
namespace optimization {

BehaviorTree::SPtr BTOptSolver::buildRelaxedBT()
{
  // Create the arena for the relaxed BT
  auto bt = std::make_shared<BehaviorTree>(std::make_unique<BehaviorTreeArena>());
  if (pModel == nullptr)
  {
    return bt;
  }

  // Create the optimization runner node and set it as entry node for the BT
  auto arena = bt->getArenaMutable();
  auto root = reinterpret_cast<RunnerOptimizer*>(
          arena->buildNode<RunnerOptimizer>("Runner_Optimizer"));

  // Set the entry node right-away
  bt->setEntryNode(root->getUniqueId());

  // Add a state domain for each variable with correspondent edge
  const auto& varsList = pModel->getVariables();
  if (varsList.empty())
  {
    return bt;
  }

  // Previous child's state nodes
  OptimizationState* prevStateNode{nullptr};
  {
    // Handle the first variable separately.
    // a) Add the selector node
    auto selector = reinterpret_cast<Selector*>(arena->buildNode<Selector>("Selector"));
    root->addChild(selector->getUniqueId());

    // b) Add the state node
    auto stateNode = reinterpret_cast<OptimizationState*>(
            arena->buildNode<OptimizationState>("Optimization_State"));
    selector->addChild(stateNode->getUniqueId());

    // Create a new edge between the selector and the state node
    auto domainEdge = arena->buildEdge(selector, stateNode);

    // Add the lower and upper bounds.
    // If lower != upper, this single edge represents a "parallel" edge.
    // In other words, to avoid creating an edge per domain element,
    // a single edge representing the whole domain is set
    auto domain = varsList[0]->getDomainMutable();
    domainEdge->setDomainBounds(domain->minElement(), domain->maxElement());

    // Add the state node to the list
    prevStateNode = stateNode;
  }

  // Handle all other variables in the model
  for (int idx{1}; idx < static_cast<int>(varsList.size()); ++idx)
  {
    // a) Add the selector node
    auto selector = reinterpret_cast<Selector*>(arena->buildNode<Selector>("Selector"));
    root->addChild(selector->getUniqueId());

    // b) Add the sequence node
    auto sequence = reinterpret_cast<Sequence*>(arena->buildNode<Sequence>("Sequence"));
    auto sequenceEdge = arena->buildEdge(selector, sequence);
    selector->addChild(sequence->getUniqueId());

    // c) Add the optimization state condition
    //    paired with the previous state node
    auto condition = reinterpret_cast<OptimizationStateCondition*>(
            arena->buildNode<OptimizationStateCondition>("State_Condition"));
    condition->pairWithOptimizationState(prevStateNode->getUniqueId());
    auto conditionEdge = arena->buildEdge(sequence, condition);
    sequence->addChild(condition->getUniqueId());

    // d) Add the state node
    auto stateNode = reinterpret_cast<OptimizationState*>(
            arena->buildNode<OptimizationState>("Optimization_State"));
    stateNode->setParentConditionNode(condition->getUniqueId());
    sequence->addChild(stateNode->getUniqueId());

    // Create a new edge between the selector and the state node
    auto domainEdge = arena->buildEdge(selector, stateNode);

    // Add the lower and upper bounds.
    // If lower != upper, this single edge represents a "parallel" edge.
    // In other words, to avoid creating an edge per domain element,
    // a single edge representing the whole domain is set
    auto domain = varsList[idx]->getDomainMutable();
    domainEdge->setDomainBounds(domain->minElement(), domain->maxElement());

    // Update previous node pointer
    prevStateNode = stateNode;
  }

  return bt;
}

void BTOptSolver::solve(uint32_t numSolutions)
{
  if (pBehaviorTree == nullptr)
  {
    return;
  }

  uint32_t solutionCtr{0};
  auto status = NodeStatus::kSuccess;
  while (status != NodeStatus::kFail)
  {
    pBehaviorTree->run();
    status = pBehaviorTree->getStatus();
    if (numSolutions > 0)
    {
      if (++solutionCtr >= numSolutions) break;
    }
  }

  // Last states collect the bounds on the solution cost
  auto blackboard = pBehaviorTree->getBlackboard();
  auto finalStates = blackboard->getOptimizationQueueMutable()->queue;

  // Choose the best final state
  OptimizationState* bestState{nullptr};
  double bestObjectiveValue = pModel->maximization() ?
          std::numeric_limits<double>::lowest() : std::numeric_limits<double>::max();
  for (auto state : finalStates)
  {
    auto optState = reinterpret_cast<OptimizationState*>(state);
    if (pModel->minimization() && optState->getGlbLowerBoundOnCost() < bestObjectiveValue)
    {
      bestState = optState;
      bestObjectiveValue = optState->getGlbLowerBoundOnCost();
    }
    else if (pModel->maximization() && optState->getGlbUpperBoundOnCost() > bestObjectiveValue)
    {
      bestState = optState;
      bestObjectiveValue = optState->getGlbUpperBoundOnCost();
    }
  }

  if (bestState == nullptr)
  {
    std::cout << "BTOptSolver - solve: NO SOLUTION FOUND!\n";
    return;
  }

  // Follow the chain of pointers from the bestState back to the states
  // associated with the first variable to set the domains
  if (pModel->maximization())
  {
    std::cout << "Cost: " << bestState->getGlbUpperBoundOnCost() << std::endl;
  }
  else
  {
    std::cout << "Cost: " << bestState->getGlbLowerBoundOnCost() << std::endl;
  }

  std::vector<std::pair<std::string, std::string>> solution;
  const auto& modelVars = pModel->getVariables();
  auto arena = pBehaviorTree->getArenaMutable();

  int ctr{0};
  for (auto it = modelVars.rbegin(); it != modelVars.rend(); ++it)
  {
    ctr++;
    auto inEdge = bestState->getIncomingEdge();
    if (inEdge == std::numeric_limits<uint32_t>::max())
    {
      throw std::runtime_error("BTOptSolver - solve: missing incoming edge");
    }
    auto edge = arena->getEdge(inEdge);
    solution.push_back({(*it)->getName(),
      std::string("[") + std::to_string(edge->getDomainLowerBound()) + ", " +
      std::to_string(edge->getDomainUpperBound()) + "]"});
    if (bestState->getParentConditionNode() == std::numeric_limits<uint32_t>::max())
    {
      // Only the first child shouldn't have parent condition nodes
      if(ctr == static_cast<int>(modelVars.size()))
      {
        break;
      }
      else
      {
        throw std::runtime_error("BTOptSolver - solve: no parent condition node");
      }
    }
    auto parentCondition = reinterpret_cast<OptimizationStateCondition*>(
            arena->getNode(bestState->getParentConditionNode()));

    // Get the list of paired states to walk backwards.
    // If more than one state is present, it means that there are multiple solutions.
    // Pick one, e.g., the first one
    const auto& pairedPreviousStatesList = parentCondition->getPairedStatesList();
    if (pairedPreviousStatesList.empty())
    {
      throw std::runtime_error("BTOptSolver - solve: no paired state with current condition node");
    }
    bestState = pairedPreviousStatesList.front();
  }

  // Print the solution
  for (auto it = solution.rbegin(); it != solution.rend(); ++it)
  {
    std::cout << (*it).first << ": " << (*it).second << std::endl;
  }
}

}  // namespace optimization
}  // btsolver
