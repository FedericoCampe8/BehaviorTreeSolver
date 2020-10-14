//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD-based implementation of the AllDifferent constraint.
//

#pragma once

#include <cstdint>  // for int64_t
#include <memory>
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
  using ValuesSet = spp::sparse_hash_set<int64_t>;

 public:
  AllDifferentState(const ValuesSet& valSet, bool isDefaultState=false);
  ~AllDifferentState() = default;

  AllDifferentState(const AllDifferentState& other);
  AllDifferentState(AllDifferentState&& other);

  AllDifferentState& operator=(const AllDifferentState& other);
  AllDifferentState& operator=(AllDifferentState&& other);

  void mergeState(DPState* other) noexcept override;

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
  void updateState(const DPState* state, int64_t val) override;

  /**
   * \brief returns the cost of taking the given value.
   * \note return +INF if the value is inducing a non-admissible state
   */
  double getCostPerValue(int64_t value) override;

  /**
   * \brief returns the list of pairs <cost, value> that can be obtains
   *        from this state when following an edge with value in [lb, ub].
   * \note values that are higher than or equal the given incumbet are discarded.
   */
  std::vector<std::pair<double, int64_t>> getCostListPerValue(
          int64_t lb, int64_t ub, double incumbent) override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isMerged() const noexcept override { return pStatesList.size() > 1; }

 private:
  /// State representation:
  /// the set of values that can still be used
  spp::sparse_hash_set<int64_t> pDomain;

  /// Original set of values
  const ValuesSet* pStartValueSet{nullptr};

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
   AllDifferent(const AllDifferentState::ValuesSet& allDiffValues,
                const std::string& name="AllDifferent");

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


   /**
    * \brief returns the initial state of the DP transformation chain as a raw pointer.
    */
   DPState* getInitialDPStateRaw() noexcept override;

 private:
   /// Set of values for the AllDifferent constraint
   AllDifferentState::ValuesSet pValSet;

   /// Initial state for the DP model for the AllDifferent constraint
   AllDifferentState::SPtr pInitialDPState{nullptr};
   void enforceConstraintForNode(Node* node, Arena* arena,
                          std::vector<std::vector<Node*>>& mddRepresentation,
                          std::vector<Node*>& newNodesList) const;
};


};
