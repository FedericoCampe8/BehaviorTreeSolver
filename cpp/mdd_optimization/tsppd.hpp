//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD-based implementation of the TSP
// Pickup and Delivery constraint.
//

#pragma once

#include <cstdint>  // for int64_t
#include <memory>
#include <string>
#include <vector>

#include <sparsepp/spp.h>

#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/mdd_constraint.hpp"
#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief AllDifferent state used for the DP model
 *        encapsulating the AllDifferent constraint.
 */
class SYS_EXPORT_STRUCT TSPPDState : public DPState {
 public:
  using PickupDeliveryPairMap = spp::sparse_hash_map<int64_t, int64_t>;
  using CostMatrix = std::vector<std::vector<int64_t>>;

 public:
  TSPPDState(PickupDeliveryPairMap* pickupDeliveryMap,
             CostMatrix* costMatrix, bool isDefaultState=false);
  ~TSPPDState() = default;

  TSPPDState(const TSPPDState& other);
  TSPPDState(TSPPDState&& other);

  TSPPDState& operator=(const TSPPDState& other);
  TSPPDState& operator=(TSPPDState&& other);

  void mergeState(DPState* other) noexcept override;

  /**
   * \brief returns true if "other" is equivalent (i.e., only one can be kept) to
   *        this state. Returns false otherwise.
   *        Here, equivalence means that the two states ("this" and "other") have the
   *        same set of next reachable states BUT they can differ, for example, on
   *        their cumulative cost and path.
   */
  bool isEqual(const DPState* other) const noexcept override;

  /**
   * \brief returns true if "other" is strictly equivalent to this state.
   *        Returns false otherwise.
   *        Here there is a notion of strong equivalence meaning that the two states
   *        ("this" and "other") have the same set of next reachable state AND they
   *        MUST be equal on their cumulative cost and path.
   */
  bool isStrictlyEqual(const DPState* other) const noexcept override;

  /**
   * \brief reset this state to the default state
   */
  void resetState() noexcept override;

  /**
   * \brief clones this states and returns a pointer to the clone.
   */
  DPState* clone() const noexcept override;

  /**
   * \brief updates this state to the next state in the DP transition function
   *        obtained by applying "val" to "state"
   */
  void updateState(const DPState* fromState, int64_t val) override;

  /**
   * \brief returns the cost of taking the given value.
   * \note return +INF if the value is inducing a non-admissible state
   */
  double getCostOnValue(int64_t value) override;

  /**
   * \brief returns the list of "width" states (if any) reachable from the current state.
   * \note discard states that have a cost greater than the given incumbent.
   */
  std::vector<DPState::UPtr> nextStateList(int64_t lb, int64_t ub, double incumbent) const override;

  /**
   * \brief returns the index of the state in the input list that can be merged
   *        with this state.
   */
  uint32_t stateSelectForMerge(const std::vector<DPState::UPtr>& statesList) const override;

  std::string toString() const noexcept override;

  bool isMerged() const noexcept override { return false; }

 private:
  /// Map of pickup-delivery nodes
  PickupDeliveryPairMap* pPickupDeliveryMap{nullptr};

  /// Matrix of costs visiting cities
  CostMatrix* pCostMatrix;

  /// Set of nodes that can be still visited
  /// from this state on
  spp::sparse_hash_set<int64_t> pDomain;
};

class SYS_EXPORT_CLASS TSPPD : public MDDConstraint {
 public:
   using UPtr = std::unique_ptr<TSPPD>;
   using SPtr = std::shared_ptr<TSPPD>;

 public:
   TSPPD(const TSPPDState::PickupDeliveryPairMap& pickupDeliveryMap,
         const TSPPDState::CostMatrix& costMatrix,
         const std::string& name="TSPPD");

   virtual ~TSPPD() {}

   /// Enforces this constraint on the given MDD node
   void enforceConstraint(Arena* arena,
                          std::vector<std::vector<Node*>>& mddRepresentation,
                          std::vector<Node*>& newNodesList) const override;

   /// Applies some heuristics to select a subset of nodes in the given layer to merge
   std::vector<Node*> mergeNodeSelect(
           int layer,
           const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept override;

   /// Merges the given list of nodes and returns the representative merged node
   Node* mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept override;

   /**
    * \brief returns the lists of states that are "equal" in "stateList" according
    *        to the DPState function implemented for this constraint.
    *        Equal states are states that can be reached from different paths but that
    *        are the root of the same sub-MDD.
    */
   std::vector<std::vector<uint32_t>> calculateEqualStates(
           const std::vector<DPState::UPtr>& statesList) override { return {}; }

   /**
    * \brief returns a lists of at most "maxWidth" sublists of states.
    *        The sublist contains the nodes that can be "merged" together when relaxing the MDD.
    */
   std::vector<std::vector<uint32_t>> calculateMergeStates(
           const std::vector<DPState::UPtr>& statesList, uint32_t maxWidth) override;

   /**
    * \brief calculates and returns the cost of the given assignment.
    */
   double calculateCost(const std::vector<int64_t>& path) const override;

   /// Returns the initial DP state
   DPState::SPtr getInitialDPState() const noexcept override;

   /**
    * \brief returns the initial state of the DP transformation chain as a raw pointer.
    */
   DPState* getInitialDPStateRaw() noexcept override;

   /// Check feasibility of AllDifferent over the variables in its scope
   bool isFeasible() const noexcept override { return true; }

 private:
   /// Constraint data
   TSPPDState::PickupDeliveryPairMap pPickupDeliveryMap;
   TSPPDState::CostMatrix pCostMatrix;

   /// Initial state for the DP model for the TSPPD constraint
   TSPPDState::SPtr pInitialDPState{nullptr};
};


};
