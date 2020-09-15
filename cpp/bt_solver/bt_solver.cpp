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

}  // btsolver
