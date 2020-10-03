//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD-based implementation of the Among constraint.
//

#pragma once

#include <algorithm>  // for std::swap
#include <cstdint>    // for int64_t
#include <limits>     // for std::numeric_limits
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
class SYS_EXPORT_STRUCT AmongState : public DPState {
 public:
  AmongState(int lower, int upper, const std::vector<int64_t>& domain);
  ~AmongState() = default;

  AmongState(const AmongState& other);
  AmongState(AmongState&& other);

  AmongState& operator=(const AmongState& other);
  AmongState& operator=(AmongState&& other);

  void mergeState(DPState* other) noexcept override;

  DPState::SPtr next(int64_t domainElement, DPState* nextDPState=nullptr) const noexcept override;

  double cost(int64_t domainElement, DPState* fromState=nullptr) const noexcept override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isEqual(const DPState* other) const noexcept override;

  bool isMerged() const noexcept override { return pIsMerged; }

  int getValueCounter() const noexcept { return pValueCounter; }

 private:
  using ValuesSet = spp::sparse_hash_set<int64_t>;

 private:
  /// Lower bound on the domain values as per Among constraint
  int pLowerBound{std::numeric_limits<int>::min()};

  /// Upper bound on the domain values as per Among constraint
  int pUpperBound{std::numeric_limits<int>::max()};

  /// Domain specifying the semantics of the Among constraint
  std::vector<int64_t> pConstraintDomain;

  /// Flag indicating whether a state has been merged or not
  bool pIsMerged{false};

  /// The state of the among constraint is represented by
  /// the shortest path from root to the current node given an edge.
  /// This is used for the upper bound of the constraint
  int pValueCounter{0};
};

class SYS_EXPORT_CLASS Among : public MDDConstraint {
 public:
   using UPtr = std::unique_ptr<Among>;
   using SPtr = std::shared_ptr<Among>;

 public:
   Among(const std::string& name="Among");

   /**
    * \brief Constructor: builds a new instance of the Among constraint.
    *        The constructor takes the lower and upper bounds values on the number of
    *        values in domain that can be assigned to the variables in this constraint's scope
    */
   Among(const std::vector<int64_t>& domain,
         int lower,
         int upper,
         const std::string& name="Among");

   /// Sets the Among constraint parameters
   void setParameters(const std::vector<int64_t>& domain, int lower, int upper);

   /// Returns true if this constraint needs to run a bottom-up pass on the mdd.
   /// Returns false otherwise
   bool runsBottomUp() const noexcept override { return true; }

   /// Sets this constraint for bottom-up separation
   void setForBottomUpFiltering() noexcept override
   {
     // Simply swap lower-upper bounds
     if (pIsTopDownFilteringEnabled)
     {
       pIsTopDownFilteringEnabled = false;
       std::swap(pLowerBound, pUpperBound);
     }
   }

   /// Sets this constraint for top-down separation
   void setForTopDownFiltering() noexcept override
   {
     if (!pIsTopDownFilteringEnabled)
     {
       pIsTopDownFilteringEnabled = true;
       std::swap(pLowerBound, pUpperBound);
     }
   }

   /// Enforces this constraint on the given MDD node
   void enforceConstraint(Node* node, Arena* arena,
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

   /// Check feasibility of Among over the variables in its scope
   bool isFeasible() const noexcept override { return true; }

 private:
   /// Flag indicating whether or not top-down filtering is enabled
   bool pIsTopDownFilteringEnabled{true};

   /// Lower bound on the domain values as per Among constraint
   int pLowerBound{std::numeric_limits<int>::min()};

   /// Upper bound on the domain values as per Among constraint
   int pUpperBound{std::numeric_limits<int>::max()};

   /// Domain specifying the semantics of the Among constraint
   std::vector<int64_t> pConstraintDomain;

   /// Initial state for the DP model for the Among constraint
   AmongState::SPtr pInitialDPState{nullptr};

   /// Counts and returns the number of occurrences of constraint values in path
   int getConstraintCountForPath(const std::vector<Edge*>& path) const;

   /// This constraint cannot be enforced only by looking at neighbors.
   /// It must be run with a top-down and bottom-up pass
   void enforceConstraintTopDown(Arena* arena,
                                 std::vector<std::vector<Node*>>& mddRepresentation) const;
   void enforceConstraintBottomUp(Arena* arena,
                                  std::vector<std::vector<Node*>>& mddRepresentation) const;

   void eraseUnfeasibleSuccessors(Node* node, Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const;
   void eraseUnfeasiblePredecessors(Node* node, Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const;

};


};
