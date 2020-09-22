//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD-based implementation of the AllDifferent constraint.
//

#pragma once


#include <string>

#include <sparsepp/spp.h>

#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/mdd_constraint.hpp"
#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief AllDifferent state used for the DP model
 *        encapsulating the AllDifferent constraint.
 */
class SYS_EXPORT_STRUCT AllDifferentState : public DPState {
 public:
  AllDifferentState();
  ~AllDifferentState() = default;

  AllDifferentState(const AllDifferentState& other);
  AllDifferentState(AllDifferentState&& other);

  AllDifferentState& operator=(const AllDifferentState& other);
  AllDifferentState& operator=(AllDifferentState&& other);

  void mergeState(DPState* other) noexcept override;

  DPState::SPtr next(int64_t domainElement) const noexcept override;

  double cost(int64_t domainElement) const noexcept override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isEqual(const DPState* other) const noexcept override;

  bool isMerged() const noexcept override { return pStatesList.size() > 1; }

 private:
  using ValuesSet = spp::sparse_hash_set<int64_t>;

 private:
  // Actual state representation
  //spp::sparse_hash_set<int64_t> pElementList;

  /// List of DP states.
  /// If the MDD is exact, there will always be one value set per DP.
  /// If the MDD is relaxed, there could be more sets, one per merged state
  std::vector<ValuesSet> pStatesList;
};

class SYS_EXPORT_CLASS AllDifferent : public MDDConstraint {
 public:
   using UPtr = std::unique_ptr<AllDifferent>;
   using SPtr = std::shared_ptr<AllDifferent>;

 public:
   AllDifferent(const std::string& name="AllDifferent");

   /// Enforces this constraint on the given MDD node
   void enforceConstraint(Node* node, Arena* arena,
                          std::vector<std::vector<Node*>>& mddRepresentation) const override;

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
   /// Initial state for the DP model for the AllDifferent constraint
   AllDifferentState::SPtr pInitialDPState{nullptr};
};


};
