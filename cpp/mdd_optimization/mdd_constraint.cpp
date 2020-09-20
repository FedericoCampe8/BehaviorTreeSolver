#include "mdd_optimization/mdd_constraint.hpp"

#include <stdexcept>  // for std::invalid_argument

namespace mdd {

MDDConstraint::MDDConstraint(ConstraintType type, const std::string& name)
: Constraint(type, name)
{
}

}  // namespace mdd

