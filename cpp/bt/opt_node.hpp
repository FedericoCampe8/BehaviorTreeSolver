//
// Copyright OptiLab 2020. All rights reserved.
//
// A collection of Behavior Tree nodes implemented for
// optimization solving.
//

#pragma once

#include <limits>  // for std::numeric_limits
#include <vector>

#include "bt/behavior_tree_arena.hpp"
#include "bt/blackboard.hpp"
#include "bt/branch.hpp"
#include "bt/edge.hpp"
#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "bt_optimization/dp_model.hpp"
#include "cp/variable.hpp"
#include "system/system_export_defs.hpp"

/// Forward declarations
namespace btsolver {
namespace optimization {
class OptimizationState;
}  // namespace optimization
}  // btsolver

namespace btsolver {
namespace optimization {

/**
 * \brief An optimization state condition node is a leaf node
 *        that represents an active/non-active condition on a state.
 *        In other words, when this node is activated, it enables a
 *        path in the Behavior Tree that considers the
 *        correspondent state.
 */
class SYS_EXPORT_CLASS OptimizationStateCondition : public Node {
 public:
   using UPtr = std::unique_ptr<OptimizationStateCondition>;

 public:
   OptimizationStateCondition(const std::string& name, BehaviorTreeArena* arena,
                              Blackboard* blackboard=nullptr);

   /// Activates this state condition node.
   /// If active, it means that the BT considered the correspondent state
   /// as part of the solution
   void activate() noexcept { pIsActive = true; }

   /// Returns the list of paired states
   const std::vector<OptimizationState*>& getPairedStatesList() const noexcept
   {
     return pPairedState;
   }

   /// Pairs this state condition with a state node.
   /// Notice that multiple optimization states can lead to
   /// the same state condition
   void pairWithOptimizationState(uint32_t state);

   /// Sets the lower bound on the solution cost
   void setGlbLowerBoundOnCost(double lb) noexcept;

   /// Sets the upper bound on the solution cost
   void setGlbUpperBoundOnCost(double ub) noexcept;

   /// Returns the lower bound on the solution cost
   double getGlbLowerBoundOnCost() const noexcept { return pTotLowerBoundCost; }

   /// Returns the upper bound on the solution cost
   double getGlbUpperBoundOnCost() const noexcept { return pTotUpperBoundCost; }

 private:
   /// Flag indicating whether or not this state condition is active
   bool pIsActive{false};

   /// Paired optimization state node used to "walk back" when
   /// an (sub) optimal solution is found
   std::vector<OptimizationState*> pPairedState;

   /// Total lower bound on the solution on this edge,
   /// i.e., the sum of the costs until and with this state transition
   double pTotLowerBoundCost{std::numeric_limits<double>::max()};

   /// Upper bound on the solution on this edge,
   /// i.e., the sum of the costs until and with this state transition
   double pTotUpperBoundCost{std::numeric_limits<double>::lowest()};

   /// Run function invoked when this node is ticked
   NodeStatus runOptimizationStateConditionNode(Blackboard* blackboard);

   /// Cleanup code invoked when this node is done ticking
   void cleanupNode(Blackboard* blackboard);
};

/**
 * \brief An optimization state node is a leaf node that represents a
 *        "state" of the optimization problem w.r.t. the
 *        variables assigned and the Dynamic Programming model.
 *        This node always returns FAIL.
 *        When it ticks, it activates the correspondent state
 *        condition node (if any) and set its edge's lower/upper bound
 *        costs.
 *
 */
class SYS_EXPORT_CLASS OptimizationState : public Node {
public:
  using UPtr = std::unique_ptr<OptimizationState>;

public:
  OptimizationState(const std::string& name, BehaviorTreeArena* arena,
                    Blackboard* blackboard=nullptr);

  /// Sets the parent condition node
  void setParentConditionNode(uint32_t stateCondition) noexcept
  {
    pParentStateConditionNode = stateCondition;
  }

  /// Returns the parent condition node
  uint32_t getParentConditionNode() const noexcept { return pParentStateConditionNode; }

  /// Pairs a state condition node to be activated once this node is ticked
  void pairStateConditionNode(uint32_t stateCondition) noexcept;

  /// Resets the internal DP state
  void resetDPState(DPState::SPtr dpState) noexcept { pDPState = dpState; }

  /// Returns the internal DP State
  DPState* getDPStateMutable() const noexcept { return pDPState.get(); }

  /// Returns the lower bound on the solution cost
  double getLowerBoundOnCost() const noexcept { return pLowerBoundCost; }

  /// Returns the upper bound on the solution cost
  double getUpperBoundOnCost() const noexcept { return pUpperBoundCost; }

  /// Returns the lower bound on the solution cost
  double getGlbLowerBoundOnCost() const noexcept { return pTotLowerBoundCost; }

  /// Returns the upper bound on the solution cost
  double getGlbUpperBoundOnCost() const noexcept { return pTotUpperBoundCost; }

private:

  /// The DP state associated with this BT state
  DPState::SPtr pDPState{nullptr};

  /// Local lower bound on the solution on this edge,
  /// i.e., the cost of this state transition
  double pLowerBoundCost{std::numeric_limits<double>::max()};

  /// Local Upper bound on the solution on this edge,
  /// i.e., the cost of this state transition
  double pUpperBoundCost{std::numeric_limits<double>::lowest()};

  /// Total lower bound on the solution on this edge,
  /// i.e., the sum of the costs until and with this state transition
  double pTotLowerBoundCost{std::numeric_limits<double>::max()};

  /// Upper bound on the solution on this edge,
  /// i.e., the sum of the costs until and with this state transition
  double pTotUpperBoundCost{std::numeric_limits<double>::lowest()};

  /// State condition node that is "parent" of this node
  /// i.e., the node that once activate, allows this node to tick
  uint32_t pParentStateConditionNode{std::numeric_limits<uint32_t>::max()};

  /// Paired state condition node
  OptimizationStateCondition* pPairedStateCondition{nullptr};

  /// Run function executed at each node's tick
  NodeStatus runOptimizationStateNode(Blackboard* blackboard);
};

/**
 * \brief A RunnerOptimizer node runs through all children in order
 *        regardless of SUCCESS/FAIL.
 *        It returns ACTIVE while children are running.
 *        It always returns Status.SUCCESS after all children have completed.
 *        The optimization process is executed by running a BFS on the children node.
 *        In particular, the RunnerOptimize keeps a queue of states to activate.
 *        The queue is filled on each child visit and given over to the next child.
 */
class SYS_EXPORT_CLASS RunnerOptimizer : public Behavior {
 public:
   using UPtr = std::unique_ptr<Selector>;
   using SPtr = std::shared_ptr<Selector>;

 public:
   RunnerOptimizer(const std::string& name, BehaviorTreeArena* arena,
                   Blackboard* blackboard);

 private:
   /// Optimization queue used for BSF cost exploration
   StateOptimizationQueue::SPtr pOptimizationQueue;

   /// The run function called on ticks
   NodeStatus runOptimizer(Blackboard* blackboard);
};

}  // namespace optimization
}  // btsolver
