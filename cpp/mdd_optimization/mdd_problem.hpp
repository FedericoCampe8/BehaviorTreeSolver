//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <memory>   // for std::unique_ptr
#include <vector>

#include "mdd_optimization/mdd_constraint.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS MDDProblem {
 public:
  using UPtr = std::unique_ptr<MDDProblem>;
  using SPtr = std::shared_ptr<MDDProblem>;
  using ConstraintsList = std::vector<MDDConstraint::SPtr>;
  using VariablesList = std::vector<Variable::SPtr>;

public:
  MDDProblem() = default;

  /// Adds the given variable to the problem
  void addVariable(Variable::SPtr var);

  /// Adds the given variable to the problem
  void addConstraint(MDDConstraint::SPtr con);

  /// Returns the list of variables in the problem
  const VariablesList& getVariables() const noexcept { return pVariablesList; }

  /// Returns the list of constraints in the problem
  const ConstraintsList& getConstraints() const noexcept { return pConstraintsList; }

  /// Set problem as maximization
  void setMaximization() noexcept { pIsMaximization = true; }

  /// Set problem as minimization
  void setMinimization() noexcept { pIsMaximization = false; }

  /// Returns true if this is a maximization problem.
  /// Returns false otherwise
  bool isMaximization() const noexcept { return pIsMaximization; }

private:
  /// Maximization flag
  bool pIsMaximization{true};

  /// List of all the variables in the problem
  VariablesList pVariablesList;

  /// List of all the constraints in the problem
  ConstraintsList pConstraintsList;

};

}  // namespace mdd
