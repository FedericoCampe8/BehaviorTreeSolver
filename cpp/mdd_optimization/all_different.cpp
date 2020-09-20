#include "mdd_optimization/all_different.hpp"

#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

namespace {
constexpr int32_t kDefaultBitmapSize{32};
}  // namespace

namespace mdd {

AllDifferentState::AllDifferentState()
: DPState()
{
  pElementList.reserve(kDefaultBitmapSize);
}

AllDifferentState::AllDifferentState(const AllDifferentState& other)
{
  pElementList = other.pElementList;
}

AllDifferentState::AllDifferentState(AllDifferentState&& other)
{
  pElementList = std::move(other.pElementList);
}

AllDifferentState& AllDifferentState::operator=(const AllDifferentState& other)
{
  if (&other == this)
  {
    return *this;
  }
  pElementList = other.pElementList;
  return *this;
}

AllDifferentState& AllDifferentState::operator=(AllDifferentState&& other)
{
  if (&other == this)
  {
    return *this;
  }
  pElementList = std::move(other.pElementList);
  return *this;
}

bool AllDifferentState::isEqual(const DPState* other) const noexcept
{
  return pElementList == reinterpret_cast<const AllDifferentState*>(other)->pElementList;
}

bool AllDifferentState::isInfeasible() const noexcept
{
  return pElementList.empty();
}

DPState::SPtr AllDifferentState::next(int32_t domainElement) const noexcept
{
  auto state = std::make_shared<AllDifferentState>();
  if (std::find(pElementList.begin(), pElementList.end(), domainElement) == pElementList.end())
  {
    state->pElementList = pElementList;
    state->pElementList.insert(domainElement);
  }
  return state;
}

double AllDifferentState::cost(int32_t domainElement) const noexcept
{
  return static_cast<double>(domainElement);
}

std::string AllDifferentState::toString() const noexcept
{
  std::string out{"{"};
  if (pElementList.empty())
  {
    out += "}";
    return out;
  }

  for (auto elem : pElementList)
  {
    out += std::to_string(elem) + ", ";
  }
  out.pop_back();
  out.pop_back();
  out += "}";
  return out;
}

AllDifferent::AllDifferent(const std::string& name)
: MDDConstraint(mdd::ConstraintType::kAllDifferent, name),
  pInitialDPState(std::make_shared<AllDifferentState>())
{
}

DPState::SPtr AllDifferent::getInitialDPState() const noexcept
{
  return pInitialDPState;
}

void AllDifferent::enforceConstraint(Node* node) const
{
  if (node == nullptr)
  {
    throw std::invalid_argument("AllDifferent - enforceConstraint: empty pointer to the node");
  }

  // TODO port implementation here
}

};
