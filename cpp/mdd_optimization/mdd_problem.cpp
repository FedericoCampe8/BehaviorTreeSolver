#include "mdd_optimization/mdd_problem.hpp"

#include <stdexcept>  // for std::invalid_argument

namespace mdd {

void MDDProblem::addVariable(Variable::SPtr var)
{
  if (var == nullptr)
  {
    throw std::invalid_argument("MDDProblem - addVariable: empty pointer to variable");
  }
  pVariablesList.push_back(var);
}

void MDDProblem::addConstraint(MDDConstraint::SPtr con)
{
  if (con == nullptr)
  {
    throw std::invalid_argument("MDDProblem - addConstraint: empty pointer to constraint");
  }
  pConstraintsList.push_back(con);
}

}  // namespace mdd
