//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a CP constraint.
//

#pragma once

#include <memory>  // for std::unique_ptr
#include <string>
#include <vector>

#include "cp/variable.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace cp {

enum class ConstraintType {
  /// Models x_i = x_j
  kEquality = 0,
  /// Models x_i < x_j
  kLessThan,
  /// Models AllDifferent(x_i, x_i+1, ..., x_j)
  kAllDifferent,
  kUnspecified
};

enum class ConstraintStatus {
  /// The constraint is scheduled to be propagated
  kReady = 0,
  /// The constraint is not currently scheduled to be propagated
  kSuspended,
  /// The constraint may be safely removed from the model without affecting feasibility
  kRedundant,
};

class SYS_EXPORT_CLASS Constraint {
 public:
   using UPtr = std::unique_ptr<Constraint>;
   using SPtr = std::shared_ptr<Constraint>;
 public:
   Constraint(ConstraintType type, const std::string& name="");

   virtual ~Constraint() = default;

   /// Returns this variable's unique identifier
   uint32_t getUniqueId() const noexcept { return pConstraintId; }

   /// Returns this constraint's name
   const std::string& getName() const noexcept { return pName; }

   /// Returns this constraint's type
   ConstraintType getType() const noexcept { return pType; }

   /// Returns the status of this constraint
   ConstraintStatus getStatus() const noexcept { return pStatus; }

   /// Sets the status of this constraint
   void setStatus(ConstraintStatus status) noexcept { pStatus = status; }

   /// Sets the scope of this constraint
   void setScope(const std::vector<Variable::SPtr>& scope) noexcept;

   /// Returns the scope of this constraint
   const std::vector<Variable::SPtr>& getScope() const noexcept { return pScope; }

   /// Given the current scope, check if the constraint is feasible,
   /// i.e., check if the variables satisfy the constraint.
   /// Returns always true if the variables are not ground
   virtual bool isFeasible() const noexcept = 0;

 private:
   static uint32_t kNextID;

 private:
   /// Unique identifier for this constraint
   uint32_t pConstraintId{0};

   /// Optional constraint name
   std::string pName{};

   /// Type of this constraint
   ConstraintType pType{ConstraintType::kUnspecified};

   /// Status of this constraint
   ConstraintStatus pStatus{ConstraintStatus::kReady};

   // Scope of this constraint, i.e., the variables involved in this constraint
   std::vector<Variable::SPtr> pScope;
};

}  // namespace cp
}  // namespace btsolver
