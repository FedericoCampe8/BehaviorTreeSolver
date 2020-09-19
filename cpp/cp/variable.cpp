#include "cp/variable.hpp"

namespace btsolver {
namespace cp {

// Initialize unique identifier for variables
uint32_t Variable::kNextID = 0;

Variable::Variable(const std::string& name, int32_t lowerBound, int32_t upperBound)
: pVariableId(Variable::kNextID++),
  pName(name),
  pLowerBound(lowerBound),
  pUpperBound(upperBound)
{
}

}  // namespace cp
}  // namespace btsolver
