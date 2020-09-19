#include "bt_optimization/bt_solver.hpp"

#include <cassert>
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
    const auto domainLB = varsList[0]->getLowerBound();
    const auto domainUB = varsList[0]->getUpperBound();

    // Handle the first variable separately.
    // a) Add the selector node and set the variable domain into it
    auto topSelector = reinterpret_cast<Selector*>(arena->buildNode<Selector>("Selector"));
    auto topSelectorEdge = root->addChild(topSelector);

    assert(topSelectorEdge != nullptr);
    topSelectorEdge->setDomainBounds(domainLB, domainUB);

    // b) Add the state nodearena->buildEdge(selector, stateNode);
    auto stateNode = reinterpret_cast<OptimizationState*>(
            arena->buildNode<OptimizationState>("Optimization_State"));
    auto domainEdge = topSelector->addChild(stateNode);

    // Add the lower and upper bounds.
    // If lower != upper, this single edge represents a "parallel" edge.
    // In other words, to avoid creating an edge per domain element,
    // a single edge representing the whole domain is set
    assert(domainEdge != nullptr);
    domainEdge->setDomainBounds(domainLB, domainUB);

    // Add the state node to the list
    prevStateNode = stateNode;
  }

  // Handle all other variables in the model
  for (int idx{1}; idx < static_cast<int>(varsList.size()); ++idx)
  {
    bt->addVariableMapping(idx, varsList[idx]);
    const auto domainLB = varsList[idx]->getLowerBound();
    const auto domainUB = varsList[idx]->getUpperBound();

    // a) Add the selector node
    //      +---+
    //      | ? |
    //      +---+
    //    since multiple choices can be made for next states
    //    under this child.
    //    Add the full domain variable to it. This is the original domain
    //    and it will be used later on children
    auto topSelector = reinterpret_cast<Selector*>(arena->buildNode<Selector>("Selector"));
    auto topSelectorEdge = root->addChild(topSelector);

    assert(topSelectorEdge != nullptr);
    topSelectorEdge->setDomainBounds(domainLB, domainUB);

    // b) Add the sequence node
    //      +---+
    //      | ->|
    //      +---+
    //    since we first need to tick the condition node and,
    //    if it returns SUCCESS, we can continue with the states
    auto sequence = reinterpret_cast<Sequence*>(arena->buildNode<Sequence>("Sequence"));
    auto sequenceEdge = arena->buildEdge(topSelector, sequence);
    topSelector->addChild(sequence);

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
    condition->pairWithOptimizationState(prevStateNode);
    sequence->addChild(condition);

    // d) Add the selector node
    //      +---+
    //      | ? |
    //      +---+
    //    since the domain of the variable on this sub-tree
    //    has (generally) more than one value that can be selected
    //    during optimization
    auto domainSelector = reinterpret_cast<Selector*>(
            arena->buildNode<Selector>("Domain_Selector"));
    sequence->addChild(domainSelector);

    // e) Add the state node
    //      +---+
    //      |U_j|
    //      +---+
    //    this state encapsulates all the possible states the
    //    variable on this child can be in
    auto stateNode = reinterpret_cast<OptimizationState*>(
            arena->buildNode<OptimizationState>("Optimization_State"));
    stateNode->addParentConditionNode(condition);
    auto domainEdge = domainSelector->addChild(stateNode);

    // Create a new edge between the domain selector and the state node
    //  +---+
    //  | ? |
    //  +---+
    //   /|\  Dx_j
    //  +---+
    //  |U_j|
    //  +---+
    // Add the lower and upper bounds.
    // If lower != upper, this single edge represents a "parallel" edge.
    // In other words, to avoid creating an edge per domain element,
    // a single edge representing the whole domain is set
    assert(domainEdge != nullptr);
    domainEdge->setDomainBounds(domainLB, domainUB);

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
    auto con = std::dynamic_pointer_cast<BTConstraint>(modelConstraint);
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

void BTOptSolver::separateConstraintInBehaviorTree(BehaviorTree* bt, BTConstraint* con)
{
  auto arena = bt->getArenaMutable();

  // Keep track of the condition states to remove during the process.
  // A condition state (and all is correspondent subtree) may be removed
  // due to invalid states
  spp::sparse_hash_set<OptimizationStateCondition*> conditionStatesToRemove;

  // Proceed one variable at a time, i.e., one child at a time
  auto root =  reinterpret_cast<RunnerOptimizer*>(arena->getNode(bt->getEntryNode()));
  const auto& childrenList = root->getChildren();

  std::vector<OptimizationState*> newStatesList;
  newStatesList.reserve(kDefaultNumNewStates);
  for (uint32_t childCtr{0}; childCtr < static_cast<uint32_t>(childrenList.size()); ++childCtr)
  {
    // Clear the list of new state before processing the current child.
    // This list represents all the new states created and possibly shared under these child.
    // For example, for the AllDifferent constraint, state {1, 2} and {2, 1} belong to
    // different branches of the same child
    newStatesList.clear();

    // Skip children that are not part of the scope of the current constraint
    if (!con->isVariableInScope(bt->getVariableMutableGivenOrderingNumber(childCtr)))
    {
      continue;
    }

    // Get the direct child of the root which is selector node.
    // Under this selector it is rooted the whole BT for the current variable
    auto topSelector = childrenList[childCtr]->cast<Selector>();

    // The "topSelector" can have two types of children:
    // 1 - one or more state node -> for the first child on the far left (i.e., first variable); or
    // 2 - one or more sequence nodes: all other children
    // Process every selector child, one at a time
    for (auto selectorChild : topSelector->getChildren())
    {
      // Depending on case (1) or (2) get the selector direct parent of the state nodes
      Selector* childSelector{nullptr};
      if (childCtr == 0)
      {
        // Case (1): the top selector is the child selector itself:
        // root -> top selector -> state node
        childSelector = topSelector;
      }
      else
      {
        // Case (2) the selector is rooted under a sequence node:
        // root -> top selector -> sequence -> condition and selector -> state node
        // See step (b) and (d) of the relaxed BT construction
        auto sequenceNode = selectorChild->cast<Sequence>();

        // From the sequence, get the selector children.
        // Notice that the sequence node has/should have two children:
        // (a) a condition node; and (b) a selector node, as per construction
        const auto& sequenceNodeChildren = sequenceNode->getChildren();
        if (sequenceNodeChildren.size() != 2)
        {
          throw std::runtime_error("BTOptSolver - separateConstraintInBehaviorTree: "
                  "invalid sequence node num. children");
        }
        childSelector = sequenceNodeChildren.at(1)->cast<Selector>();
      }

      // Now "childSelector" represents the selector right before the state node:
      //     +---+
      //     |-> |
      //     +---+
      //       |
      //   +---+---+
      //   |       |
      // /---\   +---+
      // |U_i|   | ? | <--- This selector is "childSelector"
      // \---/   +---+
      //           |   <--- This edge can be a parallel edge
      //   ^     +---+
      //   |     |U_j| <--- Current state node to separate
      //   |     +---+
      //   |
      //
      // Only for variables 2, 3, ...
      //
      // It is possible to separate the single selector w.r.t. the DP transition function.
      // Pass also the next top selector (if not on the last child) since the child on the right
      // could be modified due to splitting
      Selector* nextSelector =
              (childCtr == (static_cast<uint32_t>(childrenList.size()) - 1)) ?
              nullptr :
              childrenList[childCtr+1]->cast<Selector>();
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

        auto sequenceNode = selectorChild->cast<Sequence>();
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

void BTOptSolver::processSeparationOnChild(
        Selector* currNode, Selector* nextNode, BTConstraint* con, BehaviorTree* bt,
        std::vector<OptimizationState*>& newStatesList,
        spp::sparse_hash_set<OptimizationStateCondition*>& conditionStatesToRemove)
{
  auto arena = bt->getArenaMutable();

  // Step 1: check if the given selector has children, if not return asap
  auto childrenStateList = currNode->getChildren();
  if (childrenStateList.empty())
  {
    std::cout << "NO CHILDREN ON SELECTOR WHILE SEPARATING CHILD\n";
    return;
  }

  // Step 2: check if the state node under the given selector needs to be removed.
  //         In other words, check if there is a parent condition and the condition is in the set
  //         of states to remove.
  //         This can happen, for example, if a previous DP node on a previous variable
  //         found an invalid state
  auto stateNode = childrenStateList.front()->cast<OptimizationState>();
  if (stateNode->hasParentConditionNode())
  {
    auto parentConditionList = stateNode->getParentConditionsList();
    for (auto parentCondition : parentConditionList)
    {
      if ((conditionStatesToRemove.find(parentCondition) != conditionStatesToRemove.end()) &&
              (parentCondition->getIncomingEdge()->getHead()->getUniqueId() ==
                      currNode->getIncomingEdge()->getHead()->getUniqueId()))
      {
        // The subtree activated by the parent condition needs to be removed since
        // the parent condition itself needs to be removed (i.e., it is an invalid condition).
        // Notice that the parent condition belongs subtree of "currNode" as per check above.
        // Update the set of conditions to be removed (condition are unique nodes, meaning that
        // if it is removed now, it won't be removed again)
        conditionStatesToRemove.erase(parentCondition);

        // All the children of the selector node should be removed
        std::vector<OptimizationState*> childrenToRemove;
        for (auto childNode : childrenStateList)
        {
          auto stateNode = childNode->cast<OptimizationState>();
          if (stateNode->getAllIncomingEdges().size() > 1)
          {
            // The current state that should be removed has more than one parent.
            // Remove only the edge connected to the current selector since this state
            // could still be ticked by another edge
            for (auto incomingEdge : stateNode->getAllIncomingEdges())
            {
              if (incomingEdge->getHead()->getUniqueId() == currNode->getUniqueId())
              {
                incomingEdge->removeEdgeFromNodes();
                break;
              }
            }
          }
          else
          {
            // The current state ha only one parent, remove it
            childrenToRemove.push_back(stateNode);

            // Remove also the paired condition on next variable/child since that condition
            // will never be triggered (this state won't trigger it since it is going to be
            // removed)
            if (stateNode->getPairedCondition() != nullptr)
            {
              conditionStatesToRemove.insert(stateNode->getPairedCondition());
            }
          }
        }

        // Remove all the children
        for(auto child : childrenToRemove)
        {
          removeNodeFromBT(child, arena);
        }

        // Remove the parent condition
        removeNodeFromBT(parentCondition, arena);

        // Finally, remove also the current node
        removeNodeFromBT(currNode, arena);

        // Return from processing this child
        return;
      }
    }
  }

  // Step 3: proceed with separating the children
  // Step 3.1: reset the state of the current selector to a default state
  std::vector<OptimizationState*> stateNodeList;
  stateNodeList.reserve(childrenStateList.size());
  for (auto child : childrenStateList)
  {
    auto state = child->cast<OptimizationState>();
    stateNodeList.push_back(state);
    if (!state->hasDefaultDPState())
    {
      state->setDefaultDPState();
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
      OptimizationStateCondition* parentNode{nullptr};
      for (auto parentCondition : recursiveNode->getParentConditionsList())
      {
        // Find the parent that is under the same subtree of this state node
        if (parentCondition->getIncomingEdge()->getHead()->getUniqueId() ==
                currNode->getIncomingEdge()->getHead()->getUniqueId())
        {
          parentNode = parentCondition;
          break;
        }
      }
      assert(parentNode != nullptr);

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

    DPState::SPtr currDPState{nullptr};
    if (!recursiveNode->hasParentConditionNode() && recursiveNode->hasDefaultDPState())
    {
      // The node doesn't have a parent.
      // This mean that the node belongs to the first child.
      // Moreover, The first child has a default state:
      // use this constraint's initial state to start the DP chain
      currDPState = con->getInitialDPState();
    }
    else
    {
      // Every other state -- as per recursion -- has a non-default DP state
      currDPState = recursiveNode->getDPState();
    }

    // For each incoming arc, check if the state leading to the current node
    // using the value on the arc is consistent with the DP transition function
    auto edgeList = node->getAllIncomingEdges();
    for (auto edge: edgeList)
    {
      // Get the incoming edge and the domain values.
      // Notice that if the edge is a parallel edge,
      // there will be multiple values to process
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
        // It will end-up on a new edge when the state node is split
        edge->removeElementFromDomain(startEdgeValue);

        // Calculate the next DP state, i.e., the state reachable from the current one
        // by applying the given edge/arc value
        auto newDPState = currDPState->next(startEdgeValue);

        // Check if the value leads "no-where", i.e., if the newly produced
        // DP state is infeasible
        if (newDPState->isInfeasible())
        {
          // If the newly produced DP state is infeasible, check if the removed element
          // "startEdgeValue" made the domain empty.
          // If so, the full edge should be removed altogether with the node, parent and so on.
          // Notice that the current state may have been produced by a previous split
          if (edge->isDomainEmpty())
          {
            // Remove the edge.
            // Notice that this will remove also the edge from the head and tail nodes
            arena->deleteEdge(edge->getUniqueId());
            if (node->getPairedCondition() != nullptr)
            {
              // If there is a condition paired to the current state "node", the condition
              // should be removed since this node/DP state is about to be removed and
              // it will never trigger it
              conditionStatesToRemove.insert(node->getPairedCondition());
            }

            // Remove the node and check if all the parent selectors, after removing this node
            // become parents with no children. If so, remove also the parent selectors and
            // the corresponding state condition nodes
            if (!node->hasParentConditionNode())
            {
              // If this node doesn't have parents (i.e., first variable),
              // simply delete the child.
              // Notice that when this method will return, the caller will take care
              // of removing the parent selector node, if any
              arena->deleteNode(node->getUniqueId());
            }
            else
            {
              // There is one or more parent conditions in one or more sub-trees.
              // For each sub-tree follow the edge of the current state to the parent selector
              // and check if this state node is the only child. If so, remove everything.
              // Note: copy the lists without using references since list will be modified when
              // a node is removed
              OptimizationState::ParentConditionsList parentConditionsList =
                      node->getParentConditionsList();
              Node::EdgeList incomingEdgesList = node->getAllIncomingEdges();
              assert(parentConditionsList.size() == incomingEdgesList.size());

              for (int edgeIdx{0}; edgeIdx < static_cast<int>(incomingEdgesList.size()); ++edgeIdx)
              {
                if ((incomingEdgesList[edgeIdx]->getHead()->getAllOutgoingEdges().size()) == 1)
                {
                  // The current state node "node" is the only child and it will be removed soon.
                  // Remove also is parent and related condition
                  auto parentSelector = incomingEdgesList[edgeIdx]->getHead();

                  // Get the state condition on this sub-tree
                  auto conditionEdge = (parentSelector->getIncomingEdge()->getHead()->
                          getAllOutgoingEdges()).at(0);
                  auto conditionEdgeNode = conditionEdge->getTail();

                  // Remove both condition state and parent selector
                  arena->deleteNode(conditionEdgeNode->getUniqueId());
                  arena->deleteNode(parentSelector->getUniqueId());
                }
              }

              // Now remove the current state node
              arena->deleteNode(node->getUniqueId());
            }

            // Node has been removed due to empty domain,
            // return asap
            return;
          }
        }
        else
        {
          // === NEW DP STATE IS FEASIBLE ===

          // Split the nodes if parallel edge
          // and change the domain bounds on the original edge

          // The new state is feasible.
          // Check the current node's state
          if (node->hasDefaultDPState())
          {
            // The node has a default state, set this state as its new state reachable
            // from the edge and re-insert the value on the edge (since the value was removed
            // from the edge but it is actually an admissible value)
            edge->reinsertElementInDomain(startEdgeValue);
            node->resetDPState(newDPState);
            newStatesList.push_back(node);
          }
          else
          {
            // This doesn't have a default state, i.e., has been set from a previous loop iteration
            // and hence it must be split:
            // a) check if the same DP state has been used already, if so, re-use it.
            //    For example, in AllDifferent, the state {1, 2} and {2, 1} are the same DP state
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
                //    simply connect the edges
                auto sameStateEdge = newState->getIncomingEdge();
                if (sameStateEdge->getHead() == currNode)
                {
                  // There is no need to create a new state under the same node.
                  // Just directly re-use the previous one
                  sameStateEdge->reinsertElementInDomain(startEdgeValue);
                }
                else
                {
                  /*
                  auto domainSelectorEdge = arena->buildEdge(sameStateEdge->getHead(), node);
                  domainSelectorEdge->setDomainBounds(startEdgeValue, startEdgeValue);

                  // Add this node's condition to the other node's condition
                  for (auto condition : node->getParentConditionsList())
                  {
                    newState->addParentConditionNode(condition);
                  }
                   */
                  // Here we create a new state instead of re-using the previous one
                  auto stateNode = reinterpret_cast<OptimizationState*>(
                          arena->buildNode<OptimizationState>("Optimization_State"));
                  auto theSelector = node->getIncomingEdge()->getHead();
                  auto theSequence = theSelector->getIncomingEdge()->getHead();
                  auto theCondition = theSequence->getAllOutgoingEdges().at(0)->getTail();
                  auto parentConditionNode = theCondition->cast<OptimizationStateCondition>();
                  stateNode->addParentConditionNode(parentConditionNode);
                  auto domainSelectorEdge = currNode->addChild(stateNode);
                  domainSelectorEdge->setDomainBounds(startEdgeValue, startEdgeValue);
                  stateNode->pairStateConditionNode(newState->getPairedCondition());
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

              // Set the parent condition of the split node.
              // Note: use ONLY the condition on the current sub-tree
              auto theSelector = node->getIncomingEdge()->getHead();
              auto theSequence = theSelector->getIncomingEdge()->getHead();
              auto theCondition = theSequence->getAllOutgoingEdges().at(0)->getTail();
              auto parentConditionNode = theCondition->cast<OptimizationStateCondition>();
              stateNode->addParentConditionNode(parentConditionNode);

              auto domainSelectorEdge = currNode->addChild(stateNode);
              domainSelectorEdge->setDomainBounds(startEdgeValue, startEdgeValue);
              stateNode->resetDPState(newDPState);
              newStatesList.push_back(stateNode);

              // Create a child on the right and pair with this state
              // This is done only if there is a next node (i.e., not on the last child)
              if (nextNode != nullptr)
              {
                auto newSequence = reinterpret_cast<Sequence*>(
                        arena->buildNode<Sequence>("Sequence"));
                nextNode->addChild(newSequence);

                auto newCondition = reinterpret_cast<OptimizationStateCondition*>(
                        arena->buildNode<OptimizationStateCondition>("State_Condition"));
                newCondition->pairWithOptimizationState(stateNode);
                newSequence->addChild(newCondition);

                auto newDomainSelector = reinterpret_cast<Selector*>(
                        arena->buildNode<Selector>("Domain_Selector"));
                newSequence->addChild(newDomainSelector);

                auto newStateNode = reinterpret_cast<OptimizationState*>(
                        arena->buildNode<OptimizationState>("Optimization_State"));
                newStateNode->addParentConditionNode(newCondition);
                auto newDomainEdge = newDomainSelector->addChild(newStateNode);
                auto entryEdge = nextNode->getIncomingEdge();
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
  for (auto edge : node->getAllIncomingEdges())
  {
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
  /*
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
  */
}

}  // namespace optimization
}  // btsolver
