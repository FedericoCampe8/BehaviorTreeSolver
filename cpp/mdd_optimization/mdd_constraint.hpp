//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for an MDD constraint.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::shared_ptr

#include "mdd_optimization/constraint.hpp"
#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/node.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief Class representing a constraint used with Behavior Trees solving.
 *        Constraints used with BT solvers need to provide the corresponding
 *        Dynamic Programming model.
 */
class SYS_EXPORT_CLASS MDDConstraint : public Constraint {
 public:
   using UPtr = std::unique_ptr<MDDConstraint>;
   using SPtr = std::shared_ptr<MDDConstraint>;

 public:
   MDDConstraint(ConstraintType type, const std::string& name="");

   /// Enforces this constraint on the given MDD node
   virtual void enforceConstraint(Node* node) const = 0;

   /// Returns the initial state of the DP transformation chain
   virtual DPState::SPtr getInitialDPState() const noexcept = 0;
};

}  // namespace mdd
