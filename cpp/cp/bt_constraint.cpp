#include <stdexcept>  // for std::invalid_argument
#include "bt_constraint.hpp"

namespace btsolver {
namespace optimization {

BTConstraint::BTConstraint(cp::ConstraintType type, BehaviorTreeArena* arena,
                           const std::string& name)
: Constraint(type, name),
  pArena(arena)
{
  if (pArena == nullptr)
  {
    throw std::invalid_argument("BTConstraint: empty arena");
  }
}

}  // namespace optimization
}  // namespace btsolver
