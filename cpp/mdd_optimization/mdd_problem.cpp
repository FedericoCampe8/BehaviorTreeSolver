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

}  // namespace mdd
