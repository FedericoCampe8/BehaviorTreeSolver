#include "bt_optimization/dp_model.hpp"
#include <string>

namespace btsolver {
namespace optimization {

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

DPState::SPtr DPState::next(int32_t) const noexcept
{
  return std::make_shared<DPState>();
}

double DPState::cost(int32_t domainElement) const noexcept
{
  return static_cast<double>(domainElement);
}

bool DPState::isInfeasible() const noexcept
{
  return false;
}

std::string DPState::toString() const noexcept
{
  return "";
}

bool DPState::isEqual(const DPState*) const noexcept
{
  return true;
}

}  // namespace optimization
}  // namespace btsolver
