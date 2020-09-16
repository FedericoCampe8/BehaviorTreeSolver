#include "cp/constraint.hpp"

namespace btsolver {
namespace cp {

// Initialize unique identifier for constraints
uint32_t Constraint::kNextID = 0;

Constraint::Constraint(ConstraintType type, const std::string& name)
: pConstraintId(Constraint::kNextID++),
  pName(name),
  pType(type)
{
}

void Constraint::setScope(const std::vector<Variable::SPtr>& scope) noexcept
{
  // Set the scope
  pScope = scope;

  // Register this constraint on the variable
  for (const auto& var : pScope)
  {
    //var->registerCallbackConstraint(this);
    pVarIdSet.insert(var->getUniqueId());
  }
}

bool Constraint::isVariableInScope(Variable* var) const noexcept
{
  if (var == nullptr)
  {
    return false;
  }
  return pVarIdSet.find(var->getUniqueId()) != pVarIdSet.end();
}


}  // namespace cp
}  // namespace btsolver
