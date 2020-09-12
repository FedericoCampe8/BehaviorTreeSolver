//
// Copyright OptiLab 2020. All rights reserved.
//
// CP constraints specialized for BT solvers.
//

#pragma once

#include <memory>  // for std::unique_ptr
#include <string>
#include <vector>

#include "bt/behavior_tree_arena.hpp"
#include "bt/branch.hpp"
#include "cp/constraint.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace cp {

class SYS_EXPORT_CLASS BTConstraint : public Constraint {
 public:
   using UPtr = std::unique_ptr<BTConstraint>;
   using SPtr = std::shared_ptr<BTConstraint>;

 public:
   BTConstraint(ConstraintType type, const std::string& name="")
   : Constraint(type, name)
 {
 }

   /**
    * \brief Builds and returns the Behavior Tree used to propagate
    *        this constraint.
    */
   virtual btsolver::Sequence* builBehaviorTreePropagator(BehaviorTreeArena* arena) = 0;
};

}  // namespace cp
}  // namespace btsolver
