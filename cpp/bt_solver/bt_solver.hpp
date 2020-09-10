//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the Behavior-Tree-based solver.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::unique_ptr

#include "bt/behavior_tree.hpp"
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
};

}  // namespace btsolver
