#include "cp/variable.hpp"

namespace btsolver {
namespace cp {

// Initialize unique Identifier for variables
uint32_t Variable::kNextID = 0;

Variable::Variable(const std::string& name, int32_t lowerBound, int32_t upperBound)
: pVariableId(Variable::kNextID++),
  pName(name),
  pDomain(std::make_unique<Domain<BitmapDomain>>(lowerBound, upperBound))
{
}

int32_t Variable::getValue() const noexcept
{
  return pDomain->minElement();
}

}  // namespace cp
}  // namespace btsolver
