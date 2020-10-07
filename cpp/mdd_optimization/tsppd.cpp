#include "mdd_optimization/tsppd.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

namespace mdd {

TSPPDState::TSPPDState(PickupDeliveryPairMap* pickupDeliveryMap,
                       CostMatrix* costMatrix,
                       bool isDefaultState)
: DPState(),
  pPickupDeliveryMap(pickupDeliveryMap),
  pCostMatrix(costMatrix)
{
  if (isDefaultState)
  {
    pLastNodeVisited = 0;

    // Initialize the set of nodes that can still
    // be visited from this state on.
    // Since this is the first/default state,
    // only pick-up nodes can be taken
    for (const auto& locIter : *pickupDeliveryMap)
    {
      pDomain.insert(locIter.first);
    }
  }
}

TSPPDState::TSPPDState(const TSPPDState& other)
{
  pPickupDeliveryMap = other.pPickupDeliveryMap;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pDomain = other.pDomain;
  pPath = other.pPath;
}

TSPPDState::TSPPDState(TSPPDState&& other)
{
  pPickupDeliveryMap = other.pPickupDeliveryMap;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pDomain = std::move(other.pDomain);
  pPath = std::move(other.pPath);

  other.pPickupDeliveryMap = nullptr;
  other.pCostMatrix = nullptr;
  other.pLastNodeVisited = -1;
}

TSPPDState& TSPPDState::operator=(const TSPPDState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pPickupDeliveryMap = other.pPickupDeliveryMap;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pDomain = other.pDomain;
  pPath = other.pPath;
  return *this;
}

TSPPDState& TSPPDState::operator=(TSPPDState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pPickupDeliveryMap = other.pPickupDeliveryMap;
  pCostMatrix = other.pCostMatrix;
  pLastNodeVisited = other.pLastNodeVisited;
  pDomain = std::move(other.pDomain);
  pPath = std::move(other.pPath);

  other.pPickupDeliveryMap = nullptr;
  other.pCostMatrix = nullptr;
  other.pLastNodeVisited = -1;
  return *this;
}

bool TSPPDState::isEqual(const DPState* other) const noexcept
{
  auto otherState = reinterpret_cast<const TSPPDState*>(other);
  assert(otherState != nullptr);

  // Check if both have the same set of nodes still to explore
  if (pDomain == otherState->pDomain)
  {
    // Return true only if last node on the path is the same
    // on both states
    // TODO check if state equality should be defined in terms of costs
    return pPath.back() == otherState->pPath.back();
  }
  return false;
}

bool TSPPDState::isInfeasible() const noexcept
{
  // Easy check for an infeasible state: the path is empty
  return pPath.empty();
}

bool TSPPDState::isFeasibleValue(int64_t val, double incumbent) const noexcept
{
  // First check if the new value leads to a solution worst than the incumbent.
  // If so, return asap
  const auto cost = pPath.empty() ?
          pCostMatrix->at(0).at(val) :
          pCostMatrix->at(pPath.back()).at(val);
  if (cost >= incumbent)
  {
    return false;
  }

  // Then check if the value is a delivery, if so the pick-up node
  // must have been already visited ->
  // this is done automatically by ensuring that deliveries are only available
  // IF the correspondent pick-up is visited.

  // Then check that the value is a node that has not being visited already ->
  // this is done automatically by removing visited nodes from the domain
  return true;
}

std::vector<DPState::SPtr> TSPPDState::next(int64_t lb, int64_t ub, uint64_t width,
                                            double incumbent) const noexcept
{
  std::vector<DPState::SPtr> statesList;

  // Instead of evaluating [lb, ub] values, evaluate only the values that can
  // be reached from this state
  std::vector<std::pair<double, int64_t>> costList;
  for (auto val : pDomain)
  {
    if (val < lb || val > ub)
    {
      // Skip values that are not part of the range of admissible values
      continue;
    }

    // Note that values in the domain are either:
    // - pickup nodes: always valid nodes, the domain contains only the pickup
    //                 locations that have not being visited already
    // - delivery nodes: always valid nodes, the domain contains only the deliveries
    //                   for pickup nodes that have been already visited
    if (isFeasibleValue(val, incumbent))
    {
      // Keep track of the best values found so far
      auto costVal{pCost};
      costVal += pPath.empty() ?
              pCostMatrix->at(0).at(val) :
              pCostMatrix->at(pPath.back()).at(val);
      costList.emplace_back(costVal, val);
    }
  }

  if (costList.empty())
  {
    return statesList;
  }

  // "costList" contains the list of all values that can lead to admissible new states.
  // Sort the list to return the best states
  std::sort(costList.begin(), costList.end());
  for (int idx{0}; idx < static_cast<int>(costList.size()); ++idx)
  {
    // Create "width" new states
    const auto& cost = costList.at(idx);
    if (idx < width)
    {
      statesList.push_back(std::make_shared<TSPPDState>(pPickupDeliveryMap, pCostMatrix));
      auto newState = reinterpret_cast<TSPPDState*>(statesList.back().get());
      newState->pCost = cost.first;
      newState->pPath = pPath;
      newState->pPath.push_back(cost.second);
      newState->pDomain = pDomain;

      // Remove the current value from the list of possible domains that can be taken
      newState->pDomain.erase(cost.second);

      // Check if the current node is a pick-up node.
      // If so, add the correspondent delivery node
      if (pPickupDeliveryMap->find(cost.second) != pPickupDeliveryMap->end())
      {
        newState->pDomain.insert(pPickupDeliveryMap->at(cost.second));
      }
    }
    else
    {
      // Done
      break;
    }
  }

  // Add the last state grouping all other states that are not taken (if any)
  if (costList.size() > width)
  {
    // Group all the remaining states into a single one and push it at the back
    // of the list as the "complement" state.
    // The optimizer will re-build the MDD up to the node.
    // However, the new node won't have all the outgoing edges but only the subset
    // of edges that are not part of the
    statesList.push_back(std::make_shared<TSPPDState>(pPickupDeliveryMap, pCostMatrix));
    auto newState = reinterpret_cast<TSPPDState*>(statesList.back().get());

    // The cost is set to be the lowest of the costs that can be reached from this state
    newState->pCost = pCost;

    // The path to the node remains the same (the parent path)
    // but the domain now removes nodes just visited
    newState->pPath = pPath;
    newState->pDomain = pDomain;
    for (int idx{0}; idx < static_cast<int>(costList.size()); ++idx)
    {
      if (idx < width)
      {
        newState->pDomain.erase(costList.at(idx).second);
      }
      else
      {
        break;
      }
    }
  }
  else
  {
    // No more states to add to the list, add a default nullptr.
    // Note that a final nullptr state is also added if the width is greater than
    // the actual number of reachable states
    statesList.push_back(nullptr);
  }

  return statesList;
}

DPState::SPtr TSPPDState::next(int64_t val, DPState*) const noexcept
{
  auto state = std::make_shared<TSPPDState>(pPickupDeliveryMap, pCostMatrix);
  if (pPath.empty())
  {
    state->pPath.push_back(val);
    state->pCost = pCost + (pCostMatrix->at(0)).at(val);
  }
  else
  {
    if (pPickupDeliveryMap->find(val) != pPickupDeliveryMap->end())
    {
      // "val" is a pickup node, if not already visited, visit it
      if (std::find(pPath.begin(), pPath.end(), val) == pPath.end())
      {
        state->pPath = pPath;
        state->pPath.push_back(val);
        state->pCost = pCost + (pCostMatrix->at(pPath.back())).at(val);
      }
    }
    else
    {
      // "val" is a delivery node, check if the correspondent pickup
      // has been visited already
      int64_t pickup{-1};
      for (const auto& it : *pPickupDeliveryMap)
      {
        if (it.second == val)
        {
          pickup = it.first;
          break;
        }
      }
      assert(pickup > -1);

      if (std::find(pPath.begin(), pPath.end(), pickup) != pPath.end() &&
              std::find(pPath.begin(), pPath.end(), val) == pPath.end())
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
  return pPath.empty() ?
          static_cast<double>((pCostMatrix->at(0)).at(val)) :
          static_cast<double>((pCostMatrix->at(pPath.back())).at(val));
}

const std::vector<int64_t>& TSPPDState::cumulativePath() const noexcept
{
  return pPath;
}

double TSPPDState::cumulativeCost() const noexcept
{
  return pCost;
}

void TSPPDState::mergeState(DPState* other) noexcept
{
  auto otherDPState = reinterpret_cast<const TSPPDState*>(other);
  if (this->pCost <= otherDPState->pCost)
  {
    return;
  }

  this->pCost = otherDPState->pCost;
  pPath = otherDPState->pPath;
  pDomain = otherDPState->pDomain;
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

TSPPD::TSPPD(const TSPPDState::PickupDeliveryPairMap& pickupDeliveryMap,
             const TSPPDState::CostMatrix& costMatrix,
             const std::string& name)
: MDDConstraint(mdd::ConstraintType::kTSPPD, name),
  pPickupDeliveryMap(pickupDeliveryMap),
  pCostMatrix(costMatrix),
  pInitialDPState(std::make_shared<TSPPDState>(&pPickupDeliveryMap,
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
