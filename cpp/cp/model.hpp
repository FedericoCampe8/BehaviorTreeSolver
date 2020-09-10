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

   /// Adds a variable to the model
   void addVariable(Variable::SPtr var) { pVariablesList.push_back(var); }

   /// Returns the list of variables in the model
   const std::vector<Variable::SPtr>& getVariables() const noexcept { return pVariablesList; }

 private:
   std::string pName{};

   // Variables of the model
   std::vector<Variable::SPtr> pVariablesList;
};

}  // namespace cp
}  // namespace btsolver
