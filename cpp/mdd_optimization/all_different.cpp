#include "mdd_optimization/all_different.hpp"

#include <cassert>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

namespace {
constexpr int32_t kDefaultBitmapSize{32};
}  // namespace

namespace mdd {

AllDifferentState::AllDifferentState()
: DPState()
{
}

AllDifferentState::AllDifferentState(const AllDifferentState& other)
{
  pStatesList = other.pStatesList;
}

AllDifferentState::AllDifferentState(AllDifferentState&& other)
{
  pStatesList = std::move(other.pStatesList);
}

AllDifferentState& AllDifferentState::operator=(const AllDifferentState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = other.pStatesList;
  return *this;
}

AllDifferentState& AllDifferentState::operator=(AllDifferentState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = std::move(other.pStatesList);
  return *this;
}

bool AllDifferentState::isEqual(const DPState* other) const noexcept
{
  auto otherDPState = reinterpret_cast<const AllDifferentState*>(other);

  // Check if "other" is contained in this states
  if (pStatesList.size() < otherDPState->pStatesList.size())
  {
    // Return, there is at least one state in "other" that this DP doesn't have
    return false;
  }

  // Check that all states in "other" are contained in this state
  const auto& otherList = otherDPState->pStatesList;
  for (const auto& otherSubList : otherList)
  {
    // Check if this subSet is contained in the other state subset list
    if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
    {
      // State not found
      return false;
    }
  }

  // All states are present
  return true;
}

bool AllDifferentState::isInfeasible() const noexcept
{
  return pStatesList.empty();
}

DPState::SPtr AllDifferentState::next(int64_t domainElement) const noexcept
{
  auto state = std::make_shared<AllDifferentState>();
  if (pStatesList.empty())
  {
    state->pStatesList.resize(1);
    state->pStatesList.back().insert(domainElement);
  }
  else
  {
    for (const auto& subSet : pStatesList)
    {
      // Add the new element to all the subset compatible with it
      if (std::find(subSet.begin(), subSet.end(), domainElement) == subSet.end())
      {
        state->pStatesList.push_back(subSet);
        state->pStatesList.back().insert(domainElement);
      }
    }
  }
  return state;
}

double AllDifferentState::cost(int64_t domainElement) const noexcept
{
  return static_cast<double>(domainElement);
}

void AllDifferentState::mergeState(DPState* other) noexcept
{
  if (other == nullptr)
  {
    return;
  }

  auto otherDP = reinterpret_cast<const AllDifferentState*>(other);
  for (const auto& otherSubList : otherDP->pStatesList)
  {
    // Check if the other sublist is already present in the current list
    // and, if not, add it
    if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
    {
      pStatesList.push_back(otherSubList);
    }
  }
}

std::string AllDifferentState::toString() const noexcept
{
  std::string out{"{"};
  if (pStatesList.empty())
  {
    out += "}";
    return out;
  }

  for (auto sublist : pStatesList)
  {
    out += "{";
    for (auto val : sublist)
    {
      out += std::to_string(val) + ", ";
    }
    out.pop_back();
    out.pop_back();
    out += "}, ";
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

std::vector<Node*> AllDifferent::mergeNodeSelect(
        int layer,
        const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept
{
  // For the all different, doesn't change much what nodes to select for merging
  std::vector<Node*> nodesToMerge;
  const auto& nodesLayer = mddRepresentation[layer];
  if (nodesLayer.size() < 2)
  {
    return nodesToMerge;
  }
  nodesToMerge.push_back(nodesLayer[0]);
  nodesToMerge.push_back(nodesLayer[1]);

  return nodesToMerge;
}

Node* AllDifferent::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
{
  assert(!nodesList.empty());
  assert(arena != nullptr);

  // For all different, merging nodes selected with the "mergeNodeSelect" means merging
  // DP states on exclusive sets of values (e.g., merging {1, 2} and {1, 3})
  // Pick one at random and set it as the DP state of the new node
  auto mergedNode = arena->buildNode(nodesList.at(0)->getLayer(), nodesList.at(0)->getVariable());
  mergedNode->resetDPState(getInitialDPState());

  for (auto node : nodesList)
  {
    // Merge all nodes DP states
    mergedNode->getDPState()->mergeState(node->getDPState());
  }
  return mergedNode;
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

  // Find all children nodes of the current node
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

  // Enforce AllDifferent constraint by splitting nodes
  for (int nodeIdx{0}; nodeIdx < static_cast<int>(children.size()); ++nodeIdx)
  {
    auto nextNode = children.at(nodeIdx);
    //std::vector<int>* available_values_tail = node->get_values();
    //std::vector<int>* available_values_head = next_node->get_values();
  }
}

};
