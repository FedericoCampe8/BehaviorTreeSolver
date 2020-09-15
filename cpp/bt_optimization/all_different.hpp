//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <string>
#include <vector>

#include "bt/behavior_tree_arena.hpp"
#include "bt_optimization/dp_model.hpp"
#include "cp/opt_constraint.hpp"
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

 protected:
  bool isEqual(const DPState* other) const noexcept override;

 private:
  // Actual state representation.
  // The first element is ALWAYS the lower bound
  std::vector<int32_t> pElementList;
};

class SYS_EXPORT_CLASS AllDifferent : public BTOptConstraint {
 public:
   using UPtr = std::unique_ptr<AllDifferent>;
   using SPtr = std::shared_ptr<AllDifferent>;

 public:
   AllDifferent(BehaviorTreeArena* arena, const std::string& name="AllDifferent");
};

}  // namespace optimization
}  // namespace btsolver
