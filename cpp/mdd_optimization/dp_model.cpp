#include "mdd_optimization/dp_model.hpp"

namespace mdd {


// Initialize unique identifier for variables
uint32_t DPState::kNextID = 0;

DPState::DPState()
: pStateId(DPState::kNextID++)
{
}

bool DPState::operator==(const DPState& other)
{
  return isEqual(&other);
}

void DPState::mergeState(DPState*) noexcept
{
  // No-op
}

DPState::SPtr DPState::next(int64_t) const noexcept
{
  return std::make_shared<DPState>();
}

double DPState::cost(int64_t domainElement) const noexcept
{
  return static_cast<double>(domainElement);
}

bool DPState::isInfeasible() const noexcept
{
  return false;
}

std::string DPState::toString() const noexcept
{
  return "{DEFAULT}";
}

bool DPState::isEqual(const DPState*) const noexcept
{
  return true;
}

}  // namespace mdd
