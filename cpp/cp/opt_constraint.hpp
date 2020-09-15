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
#include "cp/constraint.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace optimization {

class SYS_EXPORT_CLASS BTOptConstraint : public cp::Constraint {
 public:
   using UPtr = std::unique_ptr<BTOptConstraint>;
   using SPtr = std::shared_ptr<BTOptConstraint>;

 public:
   BTOptConstraint(cp::ConstraintType type, BehaviorTreeArena* arena, const std::string& name="");

 protected:
   inline BehaviorTreeArena* getArena() const noexcept { return pArena; }

 private:
   /// Arena: memory area
   BehaviorTreeArena* pArena{nullptr};
};

}  // namespace optimization
}  // namespace btsolver
