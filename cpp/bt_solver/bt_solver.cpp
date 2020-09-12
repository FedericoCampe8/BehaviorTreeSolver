#include "bt_solver/bt_solver.hpp"

#include <limits>  // for std::numeric_limits
#include <string>

#include "bt/branch.hpp"
#include "bt/cp_node.hpp"
#include "bt/edge.hpp"
#include "bt/node_status.hpp"
#include "cp/bitmap_domain.hpp"
#include "cp/domain.hpp"
#include "cp/variable.hpp"

namespace btsolver {

BehaviorTree::SPtr BTSolver::buildRelaxedBT()
{
  // Create the arena for the relaxed BT
  auto bt = std::make_shared<BehaviorTree>(std::make_unique<BehaviorTreeArena>());
  if (pModel == nullptr)
  {
    return bt;
  }

  // Create a sequence node as entry node for the BT
  auto arena = bt->getArenaMutable();
  auto root = reinterpret_cast<btsolver::Sequence*>(
          arena->buildNode<btsolver::Sequence>("Root_Sequence"));

  // Set the entry node right-away
  bt->setEntryNode(root->getUniqueId());

  // Add a state domain for each variable with correspondent edge
  uint32_t stateCtr{0};
  const std::string stateName{"State_"};
  for (auto var : pModel->getVariables())
  {
    auto stateNode = arena->buildNode<btsolver::StateNode>(stateName + std::to_string(stateCtr++));
    root->addChild(stateNode->getUniqueId());

    // Add a connecting edge between the root and the state node.
    // The constructor will automatically register the node on the head and tail nodes
    auto edge = arena->buildEdge(root, stateNode);

    // Set the domain on this edge
    edge->setDomain(var->getDomainMutable());
  }

  return bt;
}

void BTSolver::solve(uint32_t numSolutions)
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
}

void BTSolver::buildExactBT(BehaviorTree::SPtr bt)
{
  if (bt == nullptr)
  {
    return;
  }

  // Get the number of children in the BT.
  // The algorithm builds the exact BT one child at a time
  const auto entryNode = bt->getEntryNode();
  if (entryNode == std::numeric_limits<uint32_t>::max())
  {
    // No entry node, return
    return;
  }

  // The root SHOULD BE always a sequence node
  auto arena = bt->getArenaMutable();
  auto root = reinterpret_cast<btsolver::Sequence*>(arena->getNode(entryNode));

  // Process each child at a time
  const auto& rootChildren = root->getChildren();
  for (int idx = 0; idx < static_cast<int>(rootChildren.size()); ++idx)
  {
    const auto prevChild = idx == 0 ? rootChildren[0] : rootChildren[idx - 1];
    processChildForExactBTConstruction(idx, rootChildren, arena);
  }
}

void BTSolver::processChildForExactBTConstruction(int childNum,
                                                  const std::vector<uint32_t>& children,
                                                  BehaviorTreeArena* arena)
{
  auto child = arena->getNode(children[childNum]);
  if (childNum == 0)
  {
    // First child simply splits on its domain
    processFirstChildForExactBTConstruction(child, arena);
    return;
  }

  // Otherwise apply state splitting and constraint filtering.
  // The initial split is the same for all children:
  // - create a new sub-BT with a selector node as "entry node"
  // - add a sequence node for each state 's' created by the left-brother BT
  // - for each sequence node add a parent-state node linked to 's'
  // - for each sequence node add new nodes considering the domain values and constraint filtering:
  //
  //     |          |
  //   +---+      +---+
  //   | U | -->  | ? |
  //   +---+      +-+-+
  //                |
  //           +----+----+
  //           |         |
  //         +---+     +---+
  //         |-> |     |-> |
  //         +---+     +---+
  //           |         |
  //      +----+----+   ...
  //      |         |
  //    /---\      ...
  //    | P | Apply filtering
  //    \---/
  // Parent state
  // for previous
  // state 's'

  // Step 1: create the selector node that will replace the current node
  auto selector = reinterpret_cast<btsolver::Selector*>(
          arena->buildNode<btsolver::Selector>("Selector"));

  // Step 1.1: replace the previous state node with the entry selector node
  auto incomingEdge = arena->getEdge(child->getIncomingEdge());
  child->removeIncomingEdge(incomingEdge->getUniqueId());
  incomingEdge->changeTail(selector);

  // Step 1.2: by CP-BT construction, the parent of each root direct child is a sequence node.
  //           Replace the state node with the new selector node
  auto parentNode = reinterpret_cast<btsolver::Sequence*>(incomingEdge->getHead());
  parentNode->replaceChild(child->getUniqueId(), selector->getUniqueId());

  // Step 2: create a sequence node for each state node on the left brother
  auto& leftBrotherStates = arena->getBlackboard()->getMostRecentStatesList();
  std::vector<btsolver::Sequence*> sequenceNodesList;
  sequenceNodesList.reserve(leftBrotherStates.size());
  for (const auto state : leftBrotherStates)
  {
    // Step 2.1: create the sequence node
    auto sequence = reinterpret_cast<btsolver::Sequence*>(
            arena->buildNode<btsolver::Selector>("Sequence"));
    sequenceNodesList.push_back(sequence);

    // Step 2.2: add the sequence node to the selector
    selector->addChild(sequence->getUniqueId());

    // Step 2.3: add an edge between the selector and the sequence.
    //           Notice that the domain is the original domain of the variable.
    //           The domain on the edge is going to be filtered with constraint filtering.
    //           Therefore, add directly the reduced domain later
    auto edge = arena->buildEdge(selector, sequence);

    // Step 2.4: add a parent state node attached to the sequence and pair it to the
    //           corresponding left-brother state.
    //           Notice that there is no need to create an edge since there is no information
    //           related to domains but only active/non-active state information
    auto parentStateNode = reinterpret_cast<btsolver::ParentStateNode*>(
            arena->buildNode<btsolver::ParentStateNode>("ParentState"));
    parentStateNode->pairState(state);
    sequence->addChild(parentStateNode->getUniqueId());
  }

  // Clear the most recent state list.
  // It will be update when creating the state for this child
  leftBrotherStates.clear();

  // Step 3: for each sequence, create the states of this child by filtering constraints.
  //         Notice that there is not two-phase approach: SPLIT domain elements then FILTER.
  //         Instead, BT construction and FILTER is done at the same time.
  //         This leads to less states and faster BT construction
  // TODO
}

void BTSolver::processFirstChildForExactBTConstruction(Node* child, BehaviorTreeArena* arena)
{
  // The given node is replaced by a selector node with a number of state children
  // equal to the number of domain elements of the variable associated with this node.
  // Step 1: create the subtree that will replace the current node.
  //         Notice that the state list is cleared (this is the first child)
  arena->getBlackboard()->getMostRecentStatesList().clear();
  auto selector = reinterpret_cast<btsolver::Selector*>(
          arena->buildNode<btsolver::Selector>("Selector"));


  // Step 2: reset the edges.
  //         The incoming edge of "child" will point to (tail) the new selector node
  //         and be removed as incoming edge from "child".
  // Note: the input child will be an orphan
  auto incomingEdge = arena->getEdge(child->getIncomingEdge());
  child->removeIncomingEdge(incomingEdge->getUniqueId());
  incomingEdge->changeTail(selector);

  // By CP-BT construction, the parent of each root direct child is a sequence node
  auto parentNode = reinterpret_cast<btsolver::Sequence*>(incomingEdge->getHead());
  parentNode->replaceChild(child->getUniqueId(), selector->getUniqueId());

  // Step 3: create a new state node for each domain element
  auto domain = incomingEdge->getDomainMutable();
  auto& it = domain->getIterator();

  bool reuseChild{true};
  while(!it.atEnd())
  {
    Edge* edge{nullptr};
    if (reuseChild)
    {
      arena->getBlackboard()->getMostRecentStatesList().push_back(child->getUniqueId());
      selector->addChild(child->getUniqueId());
      edge = arena->buildEdge(selector, child);
      reuseChild = false;
    }
    else
    {
      auto stateNode = arena->buildNode<btsolver::StateNode>("State");
      arena->getBlackboard()->getMostRecentStatesList().push_back(child->getUniqueId());
      selector->addChild(stateNode->getUniqueId());

      // Add a connecting edge between the root and the state node.
      // The constructor will automatically register the node on the head and tail nodes
      edge = arena->buildEdge(selector, stateNode);
    }

    // Set the domain on this edge
    edge->setDomainAndOwn(new cp::Domain<cp::BitmapDomain>(it.value()));
    it.moveToNext();
  }

  // Create the last node for the last domain value
  auto stateNode = arena->buildNode<btsolver::StateNode>("State");
  arena->getBlackboard()->getMostRecentStatesList().push_back(child->getUniqueId());
  selector->addChild(stateNode->getUniqueId());
  auto edge = arena->buildEdge(selector, stateNode);
  edge->setDomainAndOwn(new cp::Domain<cp::BitmapDomain>(it.value()));

  // Reset the iterator
  it.reset();
}

}  // btsolver
