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

  /// Returns the list of "width" feasible states that can be reached from the current
  /// DP state using values in [lb, ub].
  /// It also returns, as last element of the vector, the state representing all
  /// other states that could have been taken from the current state but discarded
  /// due to maximum width.
  /// @note Returns an empty vector if no state is reachible from the current one.
  /// @note Excludes all states that have a cost greater than or equal to the given incumbent
  std::vector<DPState::SPtr> next(int64_t lb, int64_t ub, uint64_t width,
                                  double incumbent) const noexcept override;

  DPState::SPtr next(int64_t val, DPState* nextDPState=nullptr) const noexcept override;

  double cost(int64_t val, DPState* fromState=nullptr) const noexcept override;

  const std::vector<int64_t>& cumulativePath() const noexcept override;

  double cumulativeCost() const noexcept override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isEqual(const DPState* other) const noexcept override;

  bool isMerged() const noexcept override { return false; }

 private:
  using NodesList = std::vector<int64_t>;

 private:
  /// Map of pickup-delivery nodes
  PickupDeliveryPairMap* pPickupDeliveryMap{nullptr};

  /// Matrix of costs visiting cities
  CostMatrix* pCostMatrix;

  /// Last node visited, i.e., this state
  mutable int64_t pLastNodeVisited{-1};

  /// Cost of the path up to this state
  double pCost{0.0};

  /// Set of nodes that can be still visited
  /// from this state on
  spp::sparse_hash_set<int64_t> pDomain;

  /// Path taken up to this point
  NodesList pPath;

  /// Check if the value if feasible according to the current state
  /// and given incumbent
  bool isFeasibleValue(int64_t val, double incumbent) const noexcept;

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

   /// Returns the initial DP state
   DPState::SPtr getInitialDPState() const noexcept override;

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
