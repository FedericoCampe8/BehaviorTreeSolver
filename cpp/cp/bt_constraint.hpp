//
// Copyright OptiLab 2020. All rights reserved.
//
// CP constraints specialized for BT optimization solvers.
//

#pragma once

#include <memory>  // for std::unique_ptr
#include <string>
#include <vector>

#include "bt/behavior_tree_arena.hpp"
#include "bt_optimization/dp_model.hpp"
#include "cp/constraint.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace optimization {

/**
 * \brief Class representing a constraint used with Behavior Trees solving.
 *        Constraints used with BT solvers need to provide the corresponding
 *        Dynamic Programming model.
 */
class SYS_EXPORT_CLASS BTConstraint : public cp::Constraint {
 public:
   using UPtr = std::unique_ptr<BTConstraint>;
   using SPtr = std::shared_ptr<BTConstraint>;

 public:
   BTConstraint(cp::ConstraintType type, BehaviorTreeArena* arena, const std::string& name="");

   /// Returns the initial state of the DP transformation chain
   virtual DPState::SPtr getInitialDPState() const noexcept = 0;

 protected:
   inline BehaviorTreeArena* getArena() const noexcept { return pArena; }

 private:
   /// Arena: memory area
   BehaviorTreeArena* pArena{nullptr};
};

}  // namespace optimization
}  // namespace btsolver
