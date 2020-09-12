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

}  // namespace cp
}  // namespace btsolver
