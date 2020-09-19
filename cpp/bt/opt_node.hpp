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
 *        /---\
 *        |U_i|
 *        \---/
 */
class SYS_EXPORT_CLASS OptimizationStateCondition : public Node {
 public:
  using PairedStatesList = std::vector<OptimizationState*>;
  using UPtr = std::unique_ptr<OptimizationStateCondition>;

 public:
   OptimizationStateCondition(const std::string& name, BehaviorTreeArena* arena);

   /// Activates this state condition node.
   /// If active, it means that the BT considered the correspondent state
   /// as part of the solution
   void activate() noexcept { pIsActive = true; }

   /// Pairs this state condition with a state node.
   /// Notice that multiple optimization states can lead to
   /// the same state condition, i.e., more than one state
   /// can activate this condition.
   /// However, each state that activates this condition shares
   /// the same Dynamic Programming state.
   /// @note this method DOES NOT create an edge between
   ///       this and the given state node
   void pairWithOptimizationState(Node* state);

   /// Sets the lower bound on the solution cost
   void setGlbLowerBoundOnCost(double lb) noexcept;

   /// Sets the upper bound on the solution cost
   void setGlbUpperBoundOnCost(double ub) noexcept;

   /// Returns the lower bound on the solution cost
   double getGlbLowerBoundOnCost() const noexcept { return pTotLowerBoundCost; }

   /// Returns the upper bound on the solution cost
   double getGlbUpperBoundOnCost() const noexcept { return pTotUpperBoundCost; }

   /// Returns the list of paired states
   const PairedStatesList& getPairedStatesList() const noexcept { return pPairedState; }

 private:
   /// Flag indicating whether or not this state condition is active
   bool pIsActive{false};

   /// Paired optimization state node used to "walk back" when
   /// an (sub) optimal solution is found
   PairedStatesList pPairedState;

   /// Total lower bound on the solution on this edge,
   /// i.e., the sum of the costs until and with this state transition
   double pTotLowerBoundCost{std::numeric_limits<double>::max()};

   /// Upper bound on the solution on this edge,
   /// i.e., the sum of the costs until and with this state transition
   double pTotUpperBoundCost{std::numeric_limits<double>::lowest()};

   /// Run function invoked when this node is ticked
   NodeStatus runNode();

   /// Cleanup code invoked when this node is done ticking
   void cleanupNode();
};

/**
 * \brief An optimization state node is a leaf node that represents a
 *        "state" of the optimization problem w.r.t. the
 *        variables assigned and the Dynamic Programming model.
 *        This node always returns FAIL.
 *        When it ticks, it activates the correspondent state
 *        condition node (if any) and set its edge's lower/upper bound
 *        costs.
 *        +---+
 *        |U_i|
 *        +---+
 */
class SYS_EXPORT_CLASS OptimizationState : public Node {
public:
  using ParentConditionsList = std::vector<OptimizationStateCondition*>;
  using UPtr = std::unique_ptr<OptimizationState>;

public:
  OptimizationState(const std::string& name, BehaviorTreeArena* arena);

  /// Adds a parent condition node,
  /// i.e., the condition node activating this state.
  /// Notice that the same state condition can be activated by different
  /// parent conditions.
  /// Indeed, a state can be shared by different sub-trees since different
  /// paths can lead to the same state (i.e., same DP state).
  /// For example, in the AllDifferent DP model, the state
  ///   {1, 2}
  /// can be reached by
  ///   {1} -> {2}
  /// or by
  ///   {2} -> {1}
  /// In the first case, the activating condition for {1, 2} is {1} while,
  /// in the second case, the activating condition for {1, 2} (= {2, 1}) is {2}.
  /// @note there is no direct edge between parent conditions this state
  void addParentConditionNode(OptimizationStateCondition* stateCondition) noexcept
  {
    if (stateCondition != nullptr)
    {
      pParentConditionsList.push_back(stateCondition);
    }
  }

  /// Returns the parent condition node
  const ParentConditionsList& getParentConditionsList() const noexcept
  {
    return pParentConditionsList;
  }

  /// Returns whether or not this node has a parent condition activating it
  bool hasParentConditionNode() const noexcept
  {
    return !pParentConditionsList.empty();
  }

  /// Pairs a state condition node to be activated when this node is ticked
  void pairStateConditionNode(OptimizationStateCondition* stateCondition) noexcept
  {
    if (stateCondition != nullptr)
    {
      pPairedStateCondition = stateCondition;
    }
  }

  /// Returns the condition state paired (ticked) by this node (if any)
  OptimizationStateCondition* getPairedCondition() const noexcept { return pPairedStateCondition; }

  /// Resets the internal DP state
  void resetDPState(DPState::SPtr dpState) noexcept
  {
    pIsDPStateChanged = true;
    pDPState = dpState;
  }

  /// Resets the internal DP state to the default one
  void setDefaultDPState()
  {
    pIsDPStateChanged = false;
    pDPState = pDefaultDPState;
  }

  /// Returns true if the nodes contains the original/default DP state.
  /// Returns false otherwise (e.g., the DP state has been reset)
  bool hasDefaultDPState() const noexcept { return !pIsDPStateChanged; }

  /// Returns the internal DP State
  DPState::SPtr getDPState() const noexcept { return pDPState; }
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
  /// Flag indicating whether or not the original DP state has been changed
  bool pIsDPStateChanged{false};

  /// The DP state associated with this BT state
  DPState::SPtr pDPState{nullptr};

  /// The default DP state
  DPState::SPtr pDefaultDPState{nullptr};

  /// Multiple edges from different child selectors can share the same
  /// optimization state under the same child.
  /// This variable keeps track of which edge is currently activated
  uint32_t pCurrentTickedEdge{0};

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
  ParentConditionsList pParentConditionsList;

  /// Paired state condition node
  OptimizationStateCondition* pPairedStateCondition{nullptr};

  /// Run function executed at each node's tick
  NodeStatus runNode();
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
   RunnerOptimizer(const std::string& name, BehaviorTreeArena* arena);

 private:
   /// The run function called on ticks
   NodeStatus runOptimizer();
};

}  // namespace optimization
}  // btsolver
