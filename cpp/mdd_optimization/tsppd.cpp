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
  pVisitedNodesList = other.pVisitedNodesList;
}

TSPPDState::TSPPDState(TSPPDState&& other)
{
  pPickUpNodeList = other.pPickUpNodeList;
  pDeliveryNodeList = other.pDeliveryNodeList;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pVisitedNodesList = std::move(other.pVisitedNodesList);

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
  pVisitedNodesList = other.pVisitedNodesList;
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
  pVisitedNodesList = std::move(other.pVisitedNodesList);

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
  if (pVisitedNodesList.size() < otherDPState->pVisitedNodesList.size())
  {
    // Return, there is at least one state in "other" that this DP doesn't have
    return false;
  }

  // Check that all states in "other" are contained in this state
  const auto& otherList = otherDPState->pVisitedNodesList;
  for (const auto& otherSubList : otherList)
  {
    // Check if this subSet is contained in the other state subset list
    if (std::find(pVisitedNodesList.begin(), pVisitedNodesList.end(), otherSubList) ==
            pVisitedNodesList.end())
    {
      // State not found
      return false;
    }
  }

  // All states are present
  return true;
}

bool TSPPDState::isInfeasible() const noexcept
{
  return pVisitedNodesList.empty();
}

DPState::SPtr TSPPDState::next(int64_t val, DPState*) const noexcept
{
  auto state = std::make_shared<TSPPDState>(pPickUpNodeList,
                                            pDeliveryNodeList,
                                            pCostMatrix);
  if (pVisitedNodesList.empty())
  {
    state->pVisitedNodesList.resize(1);
    state->pVisitedNodesList.back().insert(val);
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
    bool isDelivery = std::find(pDeliveryNodeList->begin(), pDeliveryNodeList->end(), val) !=
            pDeliveryNodeList->end();
    for (const auto& subSet : pVisitedNodesList)
    {
      if (deliveryIdx >= 0)
      {
        // A delivery can be added ONLY IF the correspondent pick-up has been
        // already visited.
        // Get the correspondent pick-up node
        const auto pickUp = pPickUpNodeList->at(deliveryIdx);
        if (std::find(subSet.begin(), subSet.end(), pickUp) != subSet.end())
        {
          // Pickup is visited already, add the delivery to the list of visited nodes
          // First check if the node is not being already visited
          if (std::find(subSet.begin(), subSet.end(), val) == subSet.end())
          {
            state->pVisitedNodesList.push_back(subSet);
            state->pVisitedNodesList.back().insert(val);
          }
        }
      }
      else
      {
        // The value corresponds to a  pick-up node which can always be visited
        // if not already visited
        if (std::find(subSet.begin(), subSet.end(), val) == subSet.end())
        {
          state->pVisitedNodesList.push_back(subSet);
          state->pVisitedNodesList.back().insert(val);
        }
      }
    }
  }

  return state;
}

double TSPPDState::cost(int64_t val, DPState* fromState) const noexcept
{

  auto fromTSPPDState = reinterpret_cast<const TSPPDState*>(fromState);
  assert(fromTSPPDState != nullptr);

  pLastNodeVisited = val;
  return (pCostMatrix->at(fromTSPPDState->pLastNodeVisited)).at(pLastNodeVisited);
}

void TSPPDState::mergeState(DPState* other) noexcept
{
  if (other == nullptr)
  {
    return;
  }

  auto otherDP = reinterpret_cast<const TSPPDState*>(other);
  for (const auto& otherSubList : otherDP->pVisitedNodesList)
  {
    // Check if the other sublist is already present in the current list
    // and, if not, add it
    if (std::find(pVisitedNodesList.begin(), pVisitedNodesList.end(), otherSubList) ==
            pVisitedNodesList.end())
    {
      pVisitedNodesList.push_back(otherSubList);
    }
  }
}

std::string TSPPDState::toString() const noexcept
{
  std::string out{"{"};
  if (pVisitedNodesList.empty())
  {
    out += "}";
    return out;
  }

  for (const auto& sublist : pVisitedNodesList)
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

void TSPPD::enforceConstraint(Node* node, Arena* arena,
                                     std::vector<std::vector<Node*>>& mddRepresentation,
                                     std::vector<Node*>& newNodesList) const
{
}

};
