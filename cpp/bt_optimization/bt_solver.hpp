//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the Behavior-Tree-based optimization solver.
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
namespace optimization {

class SYS_EXPORT_CLASS BTOptSolver {
 public:
  using UPtr = std::unique_ptr<BTOptSolver>;
  using SPtr = std::shared_ptr<BTOptSolver>;

 public:
  BTOptSolver() = default;
  ~BTOptSolver() = default;

  /// Sets the model to solve
  void setModel(cp::Model::SPtr model) noexcept { pModel = model; }

  /// Returns the model this solver is solving
  cp::Model::SPtr getModel() const noexcept { return pModel; }

  /**
   * \brief Builds and returns a relaxed Behavior Tree
   *        to be used for optimization.
   *
   *        A relaxed optimization BT has the following structure:
   *
   *        +---+
   *        | R |
   *        +---+
   *          |
   *        +---+      | Runner Optimizer node:
   *        | * | ---> | runs all children regardless success/fail.
   *        +---+      | It uses queues for BFS node activation
   *     x_1  |  x_2
   *      +---+---+
   *      |       | [min, max]
   *    +---+   +---+
   *    | U1|   | U2|
   *    +---+   +---+
   *     Dx_1    Dx_2
   */
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

}  // namespace optimization
}  // namespace btsolver
