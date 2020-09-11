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

void BTSolver::processChildForExactBTConstruction(int child, const std::vector<uint32_t>& children,
                                                  BehaviorTreeArena* arena)
{
  if (child == 0)
  {
    // First child simply splits on its domain
    processFirstChildForExactBTConstruction(arena->getNode(children[0]), arena);
    return;
  }

  // Otherwise apply state splitting and constraint filtering
}

void BTSolver::processFirstChildForExactBTConstruction(Node* child, BehaviorTreeArena* arena)
{
  // The given node is replaced by a selector node with a number of state children
  // equal to the number of domain elements of the variable associated with this node.
  // Step 1: create the subtree that will replace the current node
  auto selector = reinterpret_cast<btsolver::Selector*>(
          arena->buildNode<btsolver::Selector>("Selector"));


  // Step 2: reset the edges.
  // Note: the input child will be an orphan
  auto incomingEdge = arena->getEdge(child->getIncomingEdge());
  incomingEdge->changeTail(selector);

  // By CP-BT construction, the parent of each root direct child is a sequence node
  auto parentNode = reinterpret_cast<btsolver::Sequence*>(incomingEdge->getHead());
  parentNode->replaceChild(child->getUniqueId(), selector->getUniqueId());

  // Step 3: create a new state node for each domain element
  auto domain = incomingEdge->getDomainMutable();
  auto& it = domain->getIterator();

  // TODO reuse the input node instead of discarding it
  while(!it.atEnd())
  {
    auto stateNode = arena->buildNode<btsolver::StateNode>("State");
    selector->addChild(stateNode->getUniqueId());

    // Add a connecting edge between the root and the state node.
    // The constructor will automatically register the node on the head and tail nodes
    auto edge = arena->buildEdge(selector, stateNode);

    // Set the domain on this edge
    edge->setDomainAndOwn(new cp::Domain<cp::BitmapDomain>(it.value()));
    it.moveToNext();
  }

  // Create the last node
  auto stateNode = arena->buildNode<btsolver::StateNode>("State");
  selector->addChild(stateNode->getUniqueId());
  auto edge = arena->buildEdge(selector, stateNode);
  edge->setDomainAndOwn(new cp::Domain<cp::BitmapDomain>(it.value()));

  // Reset the iterator
  it.reset();
}

}  // btsolver
