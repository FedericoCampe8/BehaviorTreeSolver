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

DPState::SPtr AllDifferentState::next(int64_t domainElement) const noexcept
{
  auto state = std::make_shared<AllDifferentState>();
  if (std::find(pElementList.begin(), pElementList.end(), domainElement) == pElementList.end())
  {
    state->pElementList = pElementList;
    state->pElementList.insert(domainElement);
  }
  return state;
}

double AllDifferentState::cost(int64_t domainElement) const noexcept
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

  // Find all children nodes
  std::vector<Node*> children;
  children.reserve(node->getOutEdges().size());

  // Use a set for quick lookup to avoid duplicate nodes
  spp::sparse_hash_set<uint32_t> uniqueNodes;
  for (auto edge : node->getOutEdges())
  {
    auto nextNode = edge->getHead();

    // Check that the node hasn't already been added
    if ((uniqueNodes.find(nextNode->getUniqueId()) == uniqueNodes.end()) &&
            (!nextNode->isLeaf()))
    {
      uniqueNodes.insert(nextNode->getUniqueId());
      children.push_back(nextNode);
    }
  }

  // Enforce all diff contraint by splitting nodes
  for (int nodeIdx{0}; nodeIdx < static_cast<int>(children.size()); ++nodeIdx)
  {
    auto nextNode = children.at(nodeIdx);
    //std::vector<int>* available_values_tail = node->get_values();
    //std::vector<int>* available_values_head = next_node->get_values();
  }
}

};
