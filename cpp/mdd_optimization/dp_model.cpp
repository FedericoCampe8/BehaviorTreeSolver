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

bool DPState::isValueFeasible(int64_t domainElement) const noexcept
{
  return true;
}

void DPState::resetState() noexcept
{
}

DPState* DPState::clone() const noexcept
{
  return nullptr;
}

void DPState::updateState(DPState*, int64_t)
{
}

double DPState::getCostPerValue(int64_t value)
{
  return std::numeric_limits<double>::max();
}

std::vector<std::pair<double, int64_t>> DPState::getCostListPerValue(int64_t, int64_t, double)
{
  std::vector<std::pair<double, int64_t>> vals;
  return vals;
}

void DPState::copyBaseDPState(DPState* other) const
{
  other->pCost = pCost;
  other->pPath = pPath;
  other->pIsDefault = pIsDefault;
}

std::vector<DPState::SPtr> DPState::next(int64_t, int64_t, uint64_t, double) const noexcept
{
  std::vector<DPState::SPtr> res;
  return res;
}

DPState::ReplacementNodeList DPState::next(int64_t, int64_t, double,
                                           std::vector<DPState::UPtr>*) const noexcept
{
  ReplacementNodeList rnl;
  return rnl;
}

DPState::SPtr DPState::next(int64_t, DPState*) const noexcept
{
  return std::make_shared<DPState>();
}

double DPState::cost(int64_t domainElement, DPState*) const noexcept
{
  return static_cast<double>(domainElement);
}

double DPState::cumulativeCost() const noexcept
{
  return pCost;
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
  return false;
}

}  // namespace mdd
