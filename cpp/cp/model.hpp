//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for representing a CP model.
//

#pragma once

#include <memory>  // for std::unique_ptr
#include <string>
#include <vector>

#include "cp/model.hpp"
#include "cp/constraint.hpp"
#include "cp/variable.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {
namespace cp {

class SYS_EXPORT_CLASS Model {
 public:
   using UPtr = std::unique_ptr<Model>;
   using SPtr = std::shared_ptr<Model>;

 public:
   Model(const std::string& name);

   /// Returns this model's name
   const std::string& getName() const noexcept { return pName; }

   /// Set objective direction
   void setMaximization(bool maximization=true) noexcept { pMaximizeObjective = true; }

   /// Is the optimization direction set to maximize?
   bool maximization() const noexcept { return pMaximizeObjective; }

   /// Is the optimization direction set to minimize?
   bool minimization() const noexcept { return !maximization(); }

   /// Adds a variable to the model
   void addVariable(Variable::SPtr var) { pVariablesList.push_back(var); }

   /// Returns the list of variables in the model
   const std::vector<Variable::SPtr>& getVariables() const noexcept { return pVariablesList; }

   /// Adds a constraint to the model
   void addConstraint(Constraint::SPtr con) { pConstraintList.push_back(con); }

   /// Returns the list of constraint in the model
   const std::vector<Constraint::SPtr>& getConstraints() const noexcept { return pConstraintList; }

 private:
   /// The name of this model
   std::string pName{};

   /// The objective direction
   bool pMaximizeObjective{true};

   // Variables of the model
   std::vector<Variable::SPtr> pVariablesList;

   // Constraints of the model
   std::vector<Constraint::SPtr> pConstraintList;
};

}  // namespace cp
}  // namespace btsolver
