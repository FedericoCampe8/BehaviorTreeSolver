#include "mdd_optimization/dp_model.hpp"

#include <limits>  // for std::numeric_limits

namespace mdd {


// Initialize unique identifier for variables
uint32_t DPState::kNextID = 0;

DPState::DPState()
: pStateId(DPState::kNextID++)
{
  // Default state is top-down
  setStateForTopDownFiltering(true);
}

bool DPState::operator==(const DPState& other)
{
  return isEqual(&other);
}

void DPState::mergeState(DPState*) noexcept
{
  // No-op
}

DPState::SPtr DPState::next(int64_t, DPState*) const noexcept
{
  return std::make_shared<DPState>();
}

double DPState::cost(int64_t domainElement, DPState*) const noexcept
{
  return static_cast<double>(domainElement);
}

std::vector<int64_t> DPState::cumulativePath() const noexcept
{
  return {};
}

double DPState::cumulativeCost() const noexcept
{
  return std::numeric_limits<double>::max();
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
