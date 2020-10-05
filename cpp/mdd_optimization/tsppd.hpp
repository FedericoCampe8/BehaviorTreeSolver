//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD-based implementation of the AllDifferent constraint.
//

#pragma once

#include <cstdint>  // for int64_t
#include <memory>
#include <string>
#include <vector>

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
  using NodeVisitSet = std::vector<int64_t>;
  using CostMatrix = std::vector<std::vector<int64_t>>;

 public:
  TSPPDState(NodeVisitSet* pickupNodes, NodeVisitSet* deliveryNodes,
             CostMatrix* costMatrix, bool isDefaultState=false);
  ~TSPPDState() = default;

  TSPPDState(const TSPPDState& other);
  TSPPDState(TSPPDState&& other);

  TSPPDState& operator=(const TSPPDState& other);
  TSPPDState& operator=(TSPPDState&& other);

  void mergeState(DPState* other) noexcept override;

  DPState::SPtr next(int64_t val, DPState* nextDPState=nullptr) const noexcept override;

  double cost(int64_t val, DPState* fromState=nullptr) const noexcept override;

  std::vector<int64_t> cumulativePath() const noexcept override;

  double cumulativeCost() const noexcept override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isEqual(const DPState* other) const noexcept override;

  bool isMerged() const noexcept override { return false; }

 private:
  using NodesList = std::vector<int64_t>;

 private:
  /// Pointer to the set of pickup nodes
  NodeVisitSet* pPickUpNodeList;

  /// Pointer to the set of delivery nodes
  NodeVisitSet* pDeliveryNodeList;

  /// Matrix of costs visiting cities
  CostMatrix* pCostMatrix;

  /// Last node visited, i.e., this state
  mutable int64_t pLastNodeVisited{-1};

  /// Cost of the path up to this state
  double pCost{0.0};

  /// Path taken up to this point
  NodesList pPath;
};

class SYS_EXPORT_CLASS TSPPD : public MDDConstraint {
 public:
   using UPtr = std::unique_ptr<TSPPD>;
   using SPtr = std::shared_ptr<TSPPD>;

 public:
   TSPPD(const TSPPDState::NodeVisitSet& pickupNodes,
         const TSPPDState::NodeVisitSet& deliveryNodes,
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
   TSPPDState::NodeVisitSet pPickupNodes;
   TSPPDState::NodeVisitSet pDeliveryNodes;
   TSPPDState::CostMatrix pCostMatrix;

   /// Initial state for the DP model for the TSPPD constraint
   TSPPDState::SPtr pInitialDPState{nullptr};
};


};
