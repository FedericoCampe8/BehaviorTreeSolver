//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT propagation.
//

#pragma once

#include <memory>   // for std::unique_ptr
#include <string>
#include <utility>  // for std::pair
#include <vector>

#include "bt/behavior_tree_arena.hpp"
#include "bt/branch.hpp"
#include "bt/cp_node.hpp"
#include "cp/bt_constraint.hpp"
#include "cp/domain.hpp"
#include "cp/variable.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace cp {

class SYS_EXPORT_CLASS AllDifferent : public BTConstraint {
 public:
   using UPtr = std::unique_ptr<AllDifferent>;
   using SPtr = std::shared_ptr<AllDifferent>;

 public:
   AllDifferent(BehaviorTreeArena* arena, const std::string& name="AllDifferent");

   /**
    * \brief Builds and returns the Behavior Tree used to propagate
    *        this constraint.
    */
   btsolver::Sequence* builBehaviorTreePropagator() override;

   /// Given the current scope, check if the constraint is feasible,
   /// i.e., check if the variables satisfy the constraint
   bool isFeasible() const noexcept override;

 private:
   /**
    * \brief State memory is the information about states and domains
    *        passed from variable to variable while constructing the BT.
    *        This is a list like the following:
    *        [{1, 2} -> U5,
    *         {1, 3} -> U6,
    *         {1, 4} -> U7,
    *         ...
    *         ]
    *        Every new state checks itself against that list to see if this state
    *        equivalent to a previous state. If not, it add itself to the list.
    */
   using StateMemory = std::vector<std::pair<Variable::FiniteDomain*, btsolver::StateNode*>>;

 private:
   /// Builds the first node of the AllDifferent BT
   std::pair<btsolver::Selector*, StateMemory> buildFirstNodeBT(const Variable::SPtr& var,
                                                                BehaviorTreeArena* arena);

   /// Builds the a node of the AllDifferent BT
   std::pair<btsolver::Selector*, StateMemory> buildNodeBT(const Variable::SPtr& var,
                                                           StateMemory& stateMemory,
                                                           BehaviorTreeArena* arena);
};

}  // namespace cp
}  // namespace btsolver
