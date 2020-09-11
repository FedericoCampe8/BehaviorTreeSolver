//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the Behavior-Tree-based solver.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::unique_ptr
#include <vector>

#include "bt/behavior_tree.hpp"
#include "bt/behavior_tree_arena.hpp"
#include "bt/node.hpp"
#include "cp/model.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {

class SYS_EXPORT_CLASS BTSolver {
 public:
  using UPtr = std::unique_ptr<BTSolver>;
  using SPtr = std::shared_ptr<BTSolver>;

 public:
  BTSolver() = default;
  ~BTSolver() = default;

  /// Sets the model to solve
  void setModel(cp::Model::SPtr model) noexcept { pModel = model; }

  /// Returns the model this solver is solving
  cp::Model::SPtr getModel() const noexcept { return pModel; }

  /// Builds and returns a relaxed BT
  BehaviorTree::SPtr buildRelaxedBT();

  /// Compiling and solving process.
  /// This method takes the given Behavior Tree and applies
  /// node splitting and constraint filtering,
  /// child by child until the BT is an exact BT
  /// ASSUMPTION: the given BT is NOT a general BT but it is constructed
  /// to solver CP problems!
  void buildExactBT(BehaviorTree::SPtr bt);

  /// Sets the Behavior Tree instance to run and solve
  void setBehaviorTree(BehaviorTree::SPtr bt) { pBehaviorTree = bt; }

  /// Solves the Behavior Tree trying to produce the given number of solutions.
  /// If "numSolutions" is zero, it will look for all solutions
  void solve(uint32_t numSolutions);

 private:
  /// Model to solve
  cp::Model::SPtr pModel{nullptr};

  /// The Behavior Tree instance to run and solve
  BehaviorTree::SPtr pBehaviorTree{nullptr};

  /// Process "child" during the exact BT construction.
  /// This procedure applies node splitting and constraint filtering
  void processChildForExactBTConstruction(int child, const std::vector<uint32_t>& children,
                                          BehaviorTreeArena* arena);

  /// Process the first child of the exact BT.
  /// Note: the first child does not apply filtering directly, only state splitting
  void processFirstChildForExactBTConstruction(Node* child, BehaviorTreeArena* arena);
};

}  // namespace btsolver
