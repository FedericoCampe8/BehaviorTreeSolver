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
#include "bt/cp_node.hpp"
#include "cp/constraint.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace cp {

class SYS_EXPORT_CLASS BTConstraint : public Constraint {
 public:
   using UPtr = std::unique_ptr<BTConstraint>;
   using SPtr = std::shared_ptr<BTConstraint>;

 public:
   BTConstraint(ConstraintType type, BehaviorTreeArena* arena, const std::string& name="");

   /**
    * \brief Creates the relaxed BT for this constraint.
    *        Throws std::runtime_error if the scope is empty
    */
   void buildRelaxedBT();

   /**
    * \brief Returns the internal Behavior Tree representing this constraint.
    */
   btsolver::Sequence* getSemanticBT() const noexcept { return pSemanticBT; }

   /**
    * \brief Builds and returns the COMPLETE Behavior Tree used to propagate
    *        this constraint.
    */
   virtual btsolver::Sequence* builBehaviorTreePropagator() = 0;

 protected:
   inline BehaviorTreeArena* getArena() const noexcept { return pArena; }

 private:
   /// Arena: memory area
   BehaviorTreeArena* pArena{nullptr};

   /// Behavior Tree encapsulating the semantic of this constraint
   btsolver::Sequence* pSemanticBT{nullptr};
};

}  // namespace cp
}  // namespace btsolver
