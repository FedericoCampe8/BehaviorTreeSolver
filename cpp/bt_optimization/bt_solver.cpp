#include "bt_optimization/bt_solver.hpp"

#include <limits>     // for std::numeric_limits
#include <stdexcept>  // for std::runtime_error
#include <string>

#include "bt/opt_node.hpp"
#include "bt/edge.hpp"
#include "bt/node_status.hpp"
#include "cp/bitmap_domain.hpp"
#include "cp/domain.hpp"
#include "cp/variable.hpp"

namespace {
constexpr int32_t kDefaultNumNewStates{30};
}  // namespace

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
    // Set the order on the variables
    bt->addVariableMapping(0, varsList[0]);
    auto domain = varsList[0]->getDomainMutable();

    // Handle the first variable separately.
    // a) Add the selector node and set the variable domain into it
    auto topSelector = reinterpret_cast<Selector*>(arena->buildNode<Selector>("Selector"));
    root->addChild(topSelector->getUniqueId());
    auto topSelectorEdge = arena->buildEdge(root, topSelector);
    topSelectorEdge->setDomainBounds(domain->minElement(), domain->maxElement());

    // b) Add the state nodearena->buildEdge(selector, stateNode);
    auto stateNode = reinterpret_cast<OptimizationState*>(
            arena->buildNode<OptimizationState>("Optimization_State"));
    topSelector->addChild(stateNode->getUniqueId());

    // Create a new edge between the selector and the state node
    auto domainEdge = arena->buildEdge(topSelector, stateNode);

    // Add the lower and upper bounds.
    // If lower != upper, this single edge represents a "parallel" edge.
    // In other words, to avoid creating an edge per domain element,
    // a single edge representing the whole domain is set
    domainEdge->setDomainBounds(domain->minElement(), domain->maxElement());

    // Add the state node to the list
    prevStateNode = stateNode;
  }

  // Handle all other variables in the model
  for (int idx{1}; idx < static_cast<int>(varsList.size()); ++idx)
  {
    bt->addVariableMapping(idx, varsList[idx]);
    auto domain = varsList[idx]->getDomainMutable();

    // a) Add the selector node
    //      +---+
    //      | ? |
    //      +---+
    //    since multiple choices can be made for next states
    //    under this child.
    //    Add the full domain variable to it. This is the original domain
    //    and it will be used later on children
    auto topSelector = reinterpret_cast<Selector*>(arena->buildNode<Selector>("Selector"));
    root->addChild(topSelector->getUniqueId());
    auto topSelectorEdge = arena->buildEdge(root, topSelector);
    topSelectorEdge->setDomainBounds(domain->minElement(), domain->maxElement());

    // b) Add the sequence node
    //      +---+
    //      | ->|
    //      +---+
    //    since we first need to tick the condition node and,
    //    if it returns SUCCESS, we can continue with the states
    auto sequence = reinterpret_cast<Sequence*>(arena->buildNode<Sequence>("Sequence"));
    auto sequenceEdge = arena->buildEdge(topSelector, sequence);
    topSelector->addChild(sequence->getUniqueId());

    // c) Add the optimization state condition
    //    paired with the previous state node
    //      /---\
    //      |U_i|
    //      \---/
    //   the condition evaluated by the above sequence block.
    //   If this condition returns SUCCESS, the sequence will run all the
    //   following state nodes
    auto condition = reinterpret_cast<OptimizationStateCondition*>(
            arena->buildNode<OptimizationStateCondition>("State_Condition"));
    condition->pairWithOptimizationState(prevStateNode->getUniqueId());
    auto conditionEdge = arena->buildEdge(sequence, condition);
    sequence->addChild(condition->getUniqueId());

    // d) Add the selector node
    //      +---+
    //      | ? |
    //      +---+
    //    since the domain of the variable on this sub-tree
    //    has (generally) more than one value that can be selected
    //    during optimization
    auto domainSelector = reinterpret_cast<Selector*>(
            arena->buildNode<Selector>("Domain_Selector"));
    auto domainSelectorEdge = arena->buildEdge(sequence, domainSelector);
    sequence->addChild(domainSelector->getUniqueId());

    // e) Add the state node
    //      +---+
    //      |U_j|
    //      +---+
    //    this state encapsulates all the possible states the
    //    variable on this child can be in
    auto stateNode = reinterpret_cast<OptimizationState*>(
            arena->buildNode<OptimizationState>("Optimization_State"));
    stateNode->setParentConditionNode(condition->getUniqueId());
    domainSelector->addChild(stateNode->getUniqueId());

    // Create a new edge between the domain selector and the state node
    //  +---+
    //  | ? |
    //  +---+
    //   /|\  Dx_j
    //  +---+
    //  |U_j|
    //  +---+
    auto domainEdge = arena->buildEdge(domainSelector, stateNode);

    // Add the lower and upper bounds.
    // If lower != upper, this single edge represents a "parallel" edge.
    // In other words, to avoid creating an edge per domain element,
    // a single edge representing the whole domain is set
    domainEdge->setDomainBounds(domain->minElement(), domain->maxElement());

    // Update previous node pointer
    prevStateNode = stateNode;
  }

  return bt;
}

void BTOptSolver::separateBehaviorTree(BehaviorTree::SPtr bt)
{
  if (bt == nullptr || pModel == nullptr)
  {
    return;
  }

  // Get the constraints in the model.
  // Separation will be performed on each constraint
  const auto conList = pModel->getConstraints();
  if (conList.empty())
  {
    return;
  }

  for (auto& modelConstraint : conList)
  {
    // Convert the constraint to an optimization constraint
    auto con = std::dynamic_pointer_cast<BTOptConstraint>(modelConstraint);
    if (con == nullptr)
    {
      throw std::runtime_error("BTOptSolver - separateBehaviorTree: invalid constraint downcast");
    }

    // Separate one constraint at a time
    separateConstraintInBehaviorTree(bt.get(), con.get());
  }

  // Do a final pass of the BT to check if every child has states.
  // If not, the problem doesn't have a solution
}

void BTOptSolver::separateConstraintInBehaviorTree(BehaviorTree* bt, BTOptConstraint* con)
{
  auto arena = bt->getArenaMutable();

  // Keep track of the condition states to remove during the process.
  // A condition state (and all is correspondent subtree) may be removed
  // due to invalid states
  spp::sparse_hash_set<uint32_t> conditionStatesToRemove;

  // Proceed one variable at a time, i.e., one child at a time
  auto root =  reinterpret_cast<RunnerOptimizer*>(arena->getNode(bt->getEntryNode()));
  const auto& childrenIdList = root->getChildren();
  std::vector<OptimizationState*> newStatesList;
  newStatesList.reserve(kDefaultNumNewStates);
  for (uint32_t childCtr{0}; childCtr < static_cast<uint32_t>(childrenIdList.size()); ++childCtr)
  {
    // Clear the list of new state before processing the current child
    newStatesList.clear();

    // Skip children that are not part of the scope of the current constraint
    if (!con->isVariableInScope(bt->getVariableMutableGivenOrderingNumber(childCtr)))
    {
      continue;
    }

    // Get the child which should be a selector node.
    // Under the selector it is rooted the whole BT for the current variable
    auto topSelector = reinterpret_cast<Selector*>(arena->getNode(childrenIdList[childCtr]));

    // The "topSelector" can have two types of children:
    // 1 - one or more optimization state node: if the current variable is the first in the BT; or
    // 2 - one or more sequence nodes: if the current variable is not the first one in the BT
    // Process every selector child, one at a time
    for (auto selectorChildId : topSelector->getChildren())
    {
      Selector* childSelector{nullptr};
      if (childCtr == 0)
      {
        // Case (1): the top selector is the child selector itself
        childSelector = topSelector;
      }
      else
      {
        // Case (2) the selector is rooted under a sequence node.
        // See step (b) and (d) of the relaxed BT construction
        auto sequenceNode = reinterpret_cast<Sequence*>(arena->getNode(selectorChildId));

        // From the sequence, get the selector children.
        // Notice that the sequence node has/should have two children:
        // (a) a condition node; and (b) a selector node, as per construction
        const auto& sequenceNodeChildren = sequenceNode->getChildren();
        if (sequenceNodeChildren.size() != 2)
        {
          throw std::runtime_error("BTOptSolver - separateConstraintInBehaviorTree: "
                  "invalid sequence node num. children");
        }
        childSelector = reinterpret_cast<Selector*>(arena->getNode(sequenceNodeChildren[1]));
      }

      // Now it is possible to separate the single selector w.r.t.
      // the DP transition function.
      // Pass also the next top selector (if not on the last child) since the child on the right
      // could be modified due to splitting
      Selector* nextSelector =
              (childCtr == (static_cast<uint32_t>(childrenIdList.size()) - 1)) ?
              nullptr :
              reinterpret_cast<Selector*>(arena->getNode(childrenIdList[childCtr+1]));
      processSeparationOnChild(childSelector,
                               nextSelector,
                               con,
                               bt,
                               newStatesList,
                               conditionStatesToRemove);

      // If the childSelector got deleted:
      // - if it is the first child, return no solution;
      // - if it is not the first child, delete the parent sequence and, if there are no brothers
      //   return no solution
      if (childCtr > 0)
      {
        auto sequenceNode = reinterpret_cast<Sequence*>(arena->getNode(selectorChildId));
        if (sequenceNode->getAllOutgoingEdges().empty())
        {
          removeNodeFromBT(sequenceNode, arena);
        }
      }

      if (topSelector->getAllOutgoingEdges().empty())
      {
        std::cout << "NO SOLUTIONS\n";
        return;
      }
    }

  }
}

void BTOptSolver::processSeparationOnChild(Selector* currNode,
                                           Selector* nextNode,
                                           BTOptConstraint* con,
                                           BehaviorTree* bt,
                                           std::vector<OptimizationState*>& newStatesList,
                                           spp::sparse_hash_set<uint32_t>& conditionStatesToRemove)
{
  auto arena = bt->getArenaMutable();

  // Step 1: check if the given selector has children, if not return asap
  auto childrenStateList = currNode->getChildren();
  if (childrenStateList.empty())
  {
    return;
  }

  // Step 2: check if the state node under the given selector needs to be removed.
  //         In other words, check if there is a parent condition and the condition is in the set
  //         of states to remove
  auto stateNode = reinterpret_cast<OptimizationState*>(arena->getNode(childrenStateList.front()));
  if (stateNode->hasParentConditionNode())
  {
    auto parentCondition = stateNode->getParentConditionNode();
    if (conditionStatesToRemove.find(parentCondition) != conditionStatesToRemove.end())
    {
      // All these nodes need to be removed since the condition will never be triggered.
      // Step 2.1: remove the parentCondition id from the set
      conditionStatesToRemove.erase(parentCondition);

      // Step 2.2: add the paired conditions to the set for next child
      std::vector<OptimizationState*> childrenToRemove;
      childrenToRemove.reserve(childrenStateList.size());
      for (auto childId : childrenStateList)
      {
        auto child = reinterpret_cast<OptimizationState*>(arena->getNode(childId));
        childrenToRemove.push_back(child);
        if (child->getPairedCondition() != nullptr)
        {
          conditionStatesToRemove.insert(child->getPairedCondition()->getUniqueId());
        }
      }

      // Step 2.3: remove everything under the given selector "currNode"
      for(auto child : childrenToRemove)
      {
        removeNodeFromBT(child, arena);
      }

      // Step 2.4: remove the parent condition node (and its connecting edge
      removeNodeFromBT(arena->getNode(parentCondition), arena);

      // Step 2.5: remove the input selector node
      removeNodeFromBT(currNode, arena);

      // Step 2.6: nothing else to do, return
      return;
    }
  }

  // Step 3: proceed with separating the children
  // Step 3.1: reset the state of the current selector to a default state
  std::vector<OptimizationState*> stateNodeList;
  stateNodeList.reserve(childrenStateList.size());
  for (auto childId : childrenStateList)
  {
    auto node = reinterpret_cast<OptimizationState*>(arena->getNode(childId));
    stateNodeList.push_back(node);
    if (!node->hasDefaultDPState())
    {
      node->setDefaultDPState();
    }
  }

  // Consider one child/state-node at a time.
  // Each node can be split into multiple nodes according to the DP model
  // for the current constraint
  for (auto node : stateNodeList)
  {
    // Retrieve the DP state to use as starting point for the transition function.
    // If the current node doesn't have any DP state: either go back in the chain of nodes
    // following the parent condition node or, if there is no parent condition,
    // use the constraint's default DP state.
    auto recursiveNode = node;
    while (recursiveNode->hasDefaultDPState() && recursiveNode->hasParentConditionNode())
    {
      auto parentNode = reinterpret_cast<OptimizationStateCondition*>(
              arena->getNode(recursiveNode->getParentConditionNode()));

      // The parent condition node has the same state of the paired node on the left.
      // All the paired states must have the same DP state (otherwise they wouldn't be
      // the same BT state).
      // So, pick the first one and repeat the process
      recursiveNode = parentNode->getPairedStatesList().front();
      if (recursiveNode == nullptr)
      {
        throw std::runtime_error("BTOptSolver - separateConstraintInBehaviorTree: "
                "no recursive node found");
      }
    }

    DPState::SPtr initState{nullptr};
    if (!recursiveNode->hasParentConditionNode() && recursiveNode->hasDefaultDPState())
    {
      // The node doesn't have a parent.
      // This mean that the node belongs to the first child.
      // Moreover, The first child has a default state:
      // use this constraint's initial state to start the DP chain
      initState = con->getInitialDPState();
    }
    else
    {
      // Everything else already has a non-default DP state (for example, set by some
      // previous separation on brother nodes)
      initState = recursiveNode->getDPState();
    }

    // For each incoming arc, check if the state leading to the current node
    // using the value on the arc is consistent with the DP transition function
    // of the DP model representing the current constraint
    std::vector<uint32_t> edges = node->getAllIncomingEdges();
    for (auto edgeId: edges)
    {
      // Get the incoming edge and the domain values.
      // Notice that if the edge is a parallel edge,
      // there will be multiple values to process
      auto edge = arena->getEdge(edgeId);
      auto startEdgeValue = edge->getDomainLowerBound();
      auto endEdgeValue = edge->getDomainUpperBound();
      for (; startEdgeValue <= endEdgeValue; ++startEdgeValue)
      {
        if (!edge->isElementInDomain(startEdgeValue))
        {
          // Skip holes in the domain
          continue;
        }

        // Remove the current element from the domain.
        // It will end-up on another edge (if the produced DP state is feasible)
        edge->removeElementFromDomain(startEdgeValue);

        auto newDPState = initState->next(startEdgeValue);
        //std::cout << "Processing state " << newDPState->toString() << std::endl;
        if (newDPState->isInfeasible())
        {
          // If the edge has an empty domain,
          // the full edge should be removed together with the node, parent and so on
          if (edge->isDomainEmpty())
          {
            // Remove the edge
            arena->deleteEdge(edge->getUniqueId());
            if (node->getPairedCondition() != nullptr)
            {
              conditionStatesToRemove.insert(node->getPairedCondition()->getUniqueId());
            }

            // Remove the node (check whether or not the state has a condition, meaning that
            // it is not the first child)
            bool isFirstChild = !node->hasParentConditionNode();
            auto parentCondition = node->getParentConditionNode();
            arena->deleteNode(node->getUniqueId());

            // Check if the parent selector is not the first child and there is only the
            // condition node left.
            // If this is true, remove both the condition and the selector.
            // If this is the first child and there are no states under it,
            // the problem is infeasible and the caller will return no solutions
            if (!isFirstChild && (currNode->getAllOutgoingEdges().size() < 2))
            {
              // Remove the state condition
              removeNodeFromBT(arena->getNode(parentCondition), arena);

              // Remove the selector
              removeNodeFromBT(currNode, arena);

              // Return asap
              return;
            }
          }
        }
        else
        {
          // Split the nodes if parallel edge
          // and change the domain bounds on the original edge

          // The new state is feasible.
          // Check the current node's state
          if (node->hasDefaultDPState())
          {
            // The node has a default state, set this state as its new state reachable
            // from the edge and re-insert the value on the edge
            edge->reinsertElementInDomain(startEdgeValue);
            node->resetDPState(newDPState);
            newStatesList.push_back(node);
          }
          else
          {
            // This node must be split:
            // a) check the same DP state has been used already under this child, if so, re-use it
            // b) if the DP state is a new one, create a new state node and edge. Create also
            //    a new child on the right and pair it with this new state
            bool foundSameState{false};
            for (auto newState : newStatesList)
            {
              if (newState->getDPStateMutable()->isEqual(newDPState.get()))
              {
                // There is a match!
                // a) check if it is under the same selector, if so just merge the edges.
                //    This can happen, for example, if the DP transition function returns the
                //    same state for different domain elements in the variable's domain;
                // b) if the same state is under another selector (in the same child),
                auto sameStateEdge = arena->getEdge(newState->getIncomingEdge());
                if (sameStateEdge->getHead() == currNode)
                {
                  // There is no need to create a new state under the same node.
                  // Just directly re-use the previous one
                  sameStateEdge->reinsertElementInDomain(startEdgeValue);
                }
                else
                {
                  // TODO re-use the same state.
                  // In the current implementation is very complex to re-use a state.
                  // A new one is created.
                  // The same state can be re-used when the BT structure changes to link
                  // nodes directly with edges
                  auto stateNode = reinterpret_cast<OptimizationState*>(
                          arena->buildNode<OptimizationState>("Optimization_State"));
                  stateNode->setParentConditionNode(node->getParentConditionNode());
                  currNode->addChild(stateNode->getUniqueId());
                  auto domainSelectorEdge = arena->buildEdge(currNode, stateNode);
                  domainSelectorEdge->setDomainBounds(startEdgeValue, startEdgeValue);
                  stateNode->pairStateConditionNode(newState->getPairedCondition()->getUniqueId());
                  stateNode->resetDPState(newDPState);
                }
                foundSameState = true;
                break;
              }
            }

            if (!foundSameState)
            {
              // The DPState is a new one: create a new node from scratch and add it to the list
              // of new states. Also create a new node on the next right brother
              auto stateNode = reinterpret_cast<OptimizationState*>(
                      arena->buildNode<OptimizationState>("Optimization_State"));
              stateNode->setParentConditionNode(node->getParentConditionNode());
              currNode->addChild(stateNode->getUniqueId());
              auto domainSelectorEdge = arena->buildEdge(currNode, stateNode);
              domainSelectorEdge->setDomainBounds(startEdgeValue, startEdgeValue);
              stateNode->resetDPState(newDPState);
              newStatesList.push_back(stateNode);

              // Create a child on the right and pair with this state
              // This is done only if there is a next node (i.e., not on the last child)
              if (nextNode != nullptr)
              {
                auto newSequence = reinterpret_cast<Sequence*>(
                        arena->buildNode<Sequence>("Sequence"));
                auto newSequenceEdge = arena->buildEdge(nextNode, newSequence);
                nextNode->addChild(newSequence->getUniqueId());

                auto newCondition = reinterpret_cast<OptimizationStateCondition*>(
                        arena->buildNode<OptimizationStateCondition>("State_Condition"));
                newCondition->pairWithOptimizationState(stateNode->getUniqueId());
                auto newConditionEdge = arena->buildEdge(newSequence, newCondition);
                newSequence->addChild(newCondition->getUniqueId());

                auto newDomainSelector = reinterpret_cast<Selector*>(
                        arena->buildNode<Selector>("Domain_Selector"));
                auto newDomainSelectorEdge = arena->buildEdge(newSequence, newDomainSelector);
                newSequence->addChild(newDomainSelector->getUniqueId());

                auto newStateNode = reinterpret_cast<OptimizationState*>(
                        arena->buildNode<OptimizationState>("Optimization_State"));
                newStateNode->setParentConditionNode(newCondition->getUniqueId());
                newDomainSelector->addChild(newStateNode->getUniqueId());

                auto newDomainEdge = arena->buildEdge(newDomainSelector, newStateNode);
                auto entryEdge = arena->getEdge(nextNode->getIncomingEdge());
                newDomainEdge->setDomainBounds(entryEdge->getDomainLowerBound(),
                                               entryEdge->getDomainUpperBound());
              }
            }
          }
        }
      }
    }
  }
}

void BTOptSolver::removeNodeFromBT(Node* node, BehaviorTreeArena* arena)
{
  for (auto edgeId : node->getAllIncomingEdges())
  {
    auto edge = arena->getEdge(edgeId);
    edge->removeEdgeFromNodes();
    arena->deleteEdge(edge->getUniqueId());
  }
  arena->deleteNode(node->getUniqueId());
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
  // associated with the first variable to set the domains.
  // The cost of the solution is registered with the last state and it is:
  //  bestState->getGlbUpperBoundOnCost()
  // for maximization problems, and
  //  bestState->getGlbLowerBoundOnCost()
  // for minimization problems.
  // However, it is possible to calculate it dynamically while assigning variables,
  // as done in the below code
  std::vector<std::pair<std::string, std::string>> solution;
  const auto& modelVars = pModel->getVariables();
  auto arena = pBehaviorTree->getArenaMutable();

  int ctr{0};
  double solutionCost{0.0};
  for (auto it = modelVars.rbegin(); it != modelVars.rend(); ++it)
  {
    ctr++;
    auto inEdge = bestState->getIncomingEdge();
    if (inEdge == std::numeric_limits<uint32_t>::max())
    {
      throw std::runtime_error("BTOptSolver - solve: missing incoming edge");
    }
    auto edge = arena->getEdge(inEdge);
    edge->finalizeDomain();

    // If the edge is a parallel edge, pick the value that improves the cost
    double edgeCost{0.0};
    if (edge->isParallelEdge())
    {
      auto costBounds = edge->getCostBounds();
      if (pModel->maximization())
      {
        edgeCost = costBounds.second;
      }
      else
      {
        edgeCost = costBounds.first;
      }
    }
    else
    {
      edgeCost = edge->getCostValue();
    }

    solutionCost += edgeCost;
    solution.push_back({(*it)->getName(), std::to_string(edgeCost)});
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
  std::cout << "Solution cost: " << solutionCost << std::endl;
  for (auto it = solution.rbegin(); it != solution.rend(); ++it)
  {
    std::cout << (*it).first << ": " << (*it).second << std::endl;
  }
}

}  // namespace optimization
}  // btsolver
