#include "mdd_optimization/tsppd.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

// #define DEBUG

namespace mdd {

TSPPDState::TSPPDState(NodeVisitSet* pickupNodes, NodeVisitSet* deliveryNodes,
                       CostMatrix* costMatrix, bool isDefaultState)
: DPState(),
  pPickUpNodeList(pickupNodes),
  pDeliveryNodeList(deliveryNodes),
  pCostMatrix(costMatrix)
{
  if (isDefaultState)
  {
    pLastNodeVisited = 0;
  }
}

TSPPDState::TSPPDState(const TSPPDState& other)
{
  pPickUpNodeList = other.pPickUpNodeList;
  pDeliveryNodeList = other.pDeliveryNodeList;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pPath = other.pPath;
}

TSPPDState::TSPPDState(TSPPDState&& other)
{
  pPickUpNodeList = other.pPickUpNodeList;
  pDeliveryNodeList = other.pDeliveryNodeList;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pPath = std::move(other.pPath);

  other.pPickUpNodeList = nullptr;
  other.pDeliveryNodeList = nullptr;
  other.pCostMatrix = nullptr;
  other.pLastNodeVisited = -1;
}

TSPPDState& TSPPDState::operator=(const TSPPDState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pPickUpNodeList = other.pPickUpNodeList;
  pDeliveryNodeList = other.pDeliveryNodeList;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pPath = other.pPath;
  return *this;
}

TSPPDState& TSPPDState::operator=(TSPPDState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pPickUpNodeList = other.pPickUpNodeList;
  pDeliveryNodeList = other.pDeliveryNodeList;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pPath = std::move(other.pPath);

  other.pPickUpNodeList = nullptr;
  other.pDeliveryNodeList = nullptr;
  other.pCostMatrix = nullptr;
  other.pLastNodeVisited = -1;
  return *this;
}

bool TSPPDState::isEqual(const DPState* other) const noexcept
{
  auto otherDPState = reinterpret_cast<const TSPPDState*>(other);

  // Check if "other" is contained in this states
  if (pPath.size() < otherDPState->pPath.size())
  {
    // Return, there is at least one state in "other" that this DP doesn't have
    return false;
  }

  for (int idx{0}; idx < static_cast<int>(otherDPState->pPath.size()); ++idx)
  {
    if (pPath[idx] != otherDPState->pPath.at(idx))
    {
      return false;
    }
  }

  // Paths are the same
  return true;
}

bool TSPPDState::isInfeasible() const noexcept
{
  return pPath.empty();
}

DPState::SPtr TSPPDState::next(int64_t val, DPState*) const noexcept
{
  auto state = std::make_shared<TSPPDState>(pPickUpNodeList,
                                            pDeliveryNodeList,
                                            pCostMatrix);
  if (pPath.empty())
  {
    state->pPath.push_back(val);
    state->pCost = pCost + (pCostMatrix->at(0)).at(val);
  }
  else
  {
    int deliveryIdx{-1};
    for (int didx{0}; didx < static_cast<int>(pDeliveryNodeList->size()); ++didx)
    {
      if (pDeliveryNodeList->at(didx) == val)
      {
        deliveryIdx = didx;
        break;
      }
    }

    if (deliveryIdx >= 0)
    {
      // Get the correspondent pick-up node
      const auto pickUp = pPickUpNodeList->at(deliveryIdx);

      // A delivery can be added ONLY IF the correspondent pick-up has been
      // already visited and the node has not being already visited
      if (std::find(pPath.begin(), pPath.end(), pickUp) != pPath.end() &&
              std::find(pPath.begin(), pPath.end(), val) == pPath.end())
      {
        state->pPath = pPath;
        state->pPath.push_back(val);
        state->pCost = pCost + (pCostMatrix->at(pPath.back())).at(val);
      }
    }
    else
    {
      // The value corresponds to a  pick-up node which can always be visited
      // if not already visited
      if (std::find(pPath.begin(), pPath.end(), val) == pPath.end())
      {
        state->pPath = pPath;
        state->pPath.push_back(val);
        state->pCost = pCost + (pCostMatrix->at(pPath.back())).at(val);
      }
    }
  }

  return state;
}

double TSPPDState::cost(int64_t val, DPState*) const noexcept
{
  if (pPath.empty())
  {
    return static_cast<double>((pCostMatrix->at(0)).at(val));
  }
  return static_cast<double>((pCostMatrix->at(pPath.back())).at(val));
}

std::vector<int64_t> TSPPDState::cumulativePath() const noexcept
{
  return pPath;
}

double TSPPDState::cumulativeCost() const noexcept
{
  return pCost;
}

void TSPPDState::mergeState(DPState*) noexcept
{
}

std::string TSPPDState::toString() const noexcept
{
  std::string out{"{"};
  if (pPath.empty())
  {
    out += "}";
    return out;
  }
  for (auto val : pPath)
  {
    out += std::to_string(val) + ", ";
  }
  out.pop_back();
  out.pop_back();
  out += "}, ";
  return out;
}

TSPPD::TSPPD(const TSPPDState::NodeVisitSet& pickupNodes,
             const TSPPDState::NodeVisitSet& deliveryNodes,
             const TSPPDState::CostMatrix& costMatrix,
             const std::string& name)
: MDDConstraint(mdd::ConstraintType::kTSPPD, name),
  pPickupNodes(pickupNodes),
  pDeliveryNodes(deliveryNodes),
  pCostMatrix(costMatrix),
  pInitialDPState(std::make_shared<TSPPDState>(&pPickupNodes,
                                               &pDeliveryNodes,
                                               &pCostMatrix,
                                               true))
{
}

std::vector<Node*> TSPPD::mergeNodeSelect(
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

Node* TSPPD::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
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

DPState::SPtr TSPPD::getInitialDPState() const noexcept
{
  return pInitialDPState;
}

void TSPPD::enforceConstraint(Arena* arena,
                              std::vector<std::vector<Node*>>& mddRepresentation,
                              std::vector<Node*>& newNodesList) const
{
}

};
