#include "bt_solver/bt_solver.hpp"

#include "bt/behavior_tree_arena.hpp"

namespace btsolver {

BehaviorTree::SPtr BTSolver::buildRelaxedBT()
{
  // Create the arena for the relaxed BT
  auto arena = std::make_unique<BehaviorTreeArena>();
  auto bt = std::make_shared<BehaviorTree>(std::move(arena));
}

}  // btsolver
