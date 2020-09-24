//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <string>

#include <sparsepp/spp.h>
#include "cp/bt_constraint.hpp"

#include "bt/behavior_tree_arena.hpp"
#include "bt_optimization/dp_model.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace optimization {

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

  DPState::SPtr next(int32_t domainElement) const noexcept override;

  double cost(int32_t domainElement) const noexcept override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isEqual(const DPState* other) const noexcept override;

 private:
  // Actual state representation
  spp::sparse_hash_set<int32_t> pElementList;
};

class SYS_EXPORT_CLASS AllDifferent : public BTConstraint {
 public:
   using UPtr = std::unique_ptr<AllDifferent>;
   using SPtr = std::shared_ptr<AllDifferent>;

 public:
   AllDifferent(BehaviorTreeArena* arena, const std::string& name="AllDifferent");

   /// Returns the initial DP state
   DPState::SPtr getInitialDPState() const noexcept override;

   bool isFeasible() const noexcept override { return true; }

 private:
   /// Initial state for the DP model for the AllDifferent constraint
   AllDifferentState::SPtr pInitialDPState{nullptr};
};

}  // namespace optimization
}  // namespace btsolver
