#include "cp/opt_constraint.hpp"

#include <stdexcept>  // for std::invalid_argument

namespace btsolver {
namespace optimization {

BTOptConstraint::BTOptConstraint(cp::ConstraintType type, BehaviorTreeArena* arena,
                                 const std::string& name)
: Constraint(type, name),
  pArena(arena)
{
  if (pArena == nullptr)
  {
    throw std::invalid_argument("BTOptConstraint: empty arena");
  }
}

}  // namespace optimization
}  // namespace btsolver
