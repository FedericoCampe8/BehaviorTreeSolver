//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for the Behavior-Tree-based optimization solver.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::unique_ptr
#include <vector>

#include <sparsepp/spp.h>

#include "bt/behavior_tree.hpp"
#include "bt/behavior_tree_arena.hpp"
#include "bt/branch.hpp"
#include "bt/node.hpp"
#include "cp/model.hpp"
#include "cp/opt_constraint.hpp"
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
   *        +---+      | Each child activates the correspondent condition node and
   *          |        | propagates cost bounds on edges when ticked
   *          |
   *     x_1  |     x_2    x_3
   *      +---+------+------+
   *      |          |      |
   *    +---+      +---+   ...
   *    | ? |      | ? |
   *    +---+      +---+
   *     /|\         |
   *    +---+      +---+
   *    |U_1|      | ->|
   *    +---+      +---+
   *    Dx_1         |
   *             +---+---+
   *             |       |
   *           /---\   +---+
   *           | U1|   | ? |
   *           \---/   +---+
   *                    /|\
   *                   +---+
   *                   |U_2|
   *                   +---+
   *                   Dx_2
   */
  BehaviorTree::SPtr buildRelaxedBT();

  /// Sets the Behavior Tree instance to solve.
  /// Note: the BT must be built either as a relaxed BT or after separation/top-down compilation.
  /// In other words, this method expects a valid BT for optimization
  void setBehaviorTree(BehaviorTree::SPtr bt) { pBehaviorTree = bt; }

  /// Given a (relaxed) behavior tree, this method builds the exact behavior tree
  /// w.r.t. the input model by separating constraint.
  /// For more information, see
  /// "Construction by Separation", cap. 3.4
  /// https://www.cmu.edu/tepper/programs/phd/program/assets/dissertations/2014-operations-research-cire-dissertation.pdf
  void separateBehaviorTree(BehaviorTree::SPtr bt);

  /// Solves the Behavior Tree trying to produce the given number of solutions.
  /// If "numSolutions" is zero, it will look for all solutions
  void solve(uint32_t numSolutions);

 private:
  /// Model to solve
  cp::Model::SPtr pModel{nullptr};

  /// The Behavior Tree instance to run and solve
  BehaviorTree::SPtr pBehaviorTree{nullptr};

  /// Separates the given constraint in the given BT
  void separateConstraintInBehaviorTree(BehaviorTree* bt, BTOptConstraint* con);

  /// Process the separation on the given current node
  void processSeparationOnChild(Selector* currNode,
                                Selector* nextNode,
                                BTOptConstraint* con,
                                BehaviorTree* bt,
                                std::vector<OptimizationState*>& newStatesList,
                                spp::sparse_hash_set<uint32_t>& conditionStatesToRemove);

  /// Removes a node and all its incoming edges from the BT
  void removeNodeFromBT(Node* node, BehaviorTreeArena* arena);

};

}  // namespace optimization
}  // namespace btsolver
