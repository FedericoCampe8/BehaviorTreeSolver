//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD-based implementation of the AllDifferent constraint.
//

#pragma once


#include <string>
#include <queue>

#include <sparsepp/spp.h>

#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/mdd_constraint.hpp"
#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief Among state used for the DP model
 *        encapsulating the Among constraint.
 */
class SYS_EXPORT_STRUCT AmongState : public DPState {
 public:
  AmongState();
  ~AmongState() = default;

  AmongState(const AmongState& other);
  AmongState(AmongState&& other);

  AmongState& operator=(const AmongState& other);
  AmongState& operator=(AmongState&& other);

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

class SYS_EXPORT_CLASS Among : public MDDConstraint {
 public:
   using UPtr = std::unique_ptr<Among>;
   using SPtr = std::shared_ptr<Among>;

 public:
   Among(const std::string& name="Among");

   /// Enforces this constraint on the given MDD node
   void enforceConstraint(Node* node, Arena* arena,
                          std::vector<std::vector<Node*>>& mddRepresentation, 
                          std::vector<Node*>& newNodesList) const override;


    // This constraint cannot be enforced only by looking at neighbors. We need a top-down and bottom-up pass
    //TODO Hacky way to use the common enforceConstraint method. Needs refactoring.
    void enforceConstraintTopDown(Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const ;
    void enforceConstraintBottomUp(Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const ;
    

   /// Applies some heuristics to select a subset of nodes in the given layer to merge
   std::vector<Node*> mergeNodeSelect(
           int layer,
           const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept override;

   /// Merges the given list of nodes and returns the representative merged node
   Node* mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept override;

   // Counts the number of occurrences of constraint values in path
   int getConstraintCountForPath( std::vector<Edge*> path ) const;

//    /// Returns the initial DP state
   DPState::SPtr getInitialDPState() const noexcept override;

   /// Check feasibility of AllDifferent over the variables in its scope
   bool isFeasible() const noexcept override { return true; }

   void setParameters(const std::vector<int>& domain, int lower, int upper)
   {
       pConstraintDomain = domain;
       pLowerBound = lower;
       pUpperBound = upper;
   }


 private:
      std::vector<int> pConstraintDomain;
      int pLowerBound;
      int pUpperBound;
//    /// Initial state for the DP model for the AllDifferent constraint
   AmongState::SPtr pInitialDPState{nullptr};

};


};
