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

  DPState::SPtr next(int64_t domainElement) const noexcept override;

  double cost(int64_t domainElement) const noexcept override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isEqual(const DPState* other) const noexcept override;

 private:
  // Actual state representation
  spp::sparse_hash_set<int64_t> pElementList;
};

class SYS_EXPORT_CLASS AllDifferent : public MDDConstraint {
 public:
   using UPtr = std::unique_ptr<AllDifferent>;
   using SPtr = std::shared_ptr<AllDifferent>;

 public:
   AllDifferent(const std::string& name="AllDifferent");

   /// Enforces this constraint on the given MDD node
   void enforceConstraint(Node* node) const override;

   /// Returns the initial DP state
   DPState::SPtr getInitialDPState() const noexcept override;

   /// Check feasibility of AllDifferent over the variables in its scope
   bool isFeasible() const noexcept override { return true; }

 private:
   /// Initial state for the DP model for the AllDifferent constraint
   AllDifferentState::SPtr pInitialDPState{nullptr};
};


};