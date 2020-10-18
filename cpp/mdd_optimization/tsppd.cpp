#include "mdd_optimization/tsppd.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <iostream>
#include <limits>     // for std::numeric_limits
#include <stdexcept>  // for std::invalid_argument
#include <sstream>
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
    pCost = 0.0;

    // Initialize the set of nodes that can still
    // be visited from this state on.
    // Since this is the first/default state,
    // only pick-up nodes can be taken
    for (const auto& locIter : *pPickupDeliveryMap)
    {
      pDomain.insert(locIter.first);
    }
  }
}

TSPPDState::TSPPDState(const TSPPDState& other)
{
  pPickupDeliveryMap = other.pPickupDeliveryMap;
  pCostMatrix = other.pCostMatrix;
  pDomain = other.pDomain;
  pPath = other.pPath;
  pIsExact = other.pIsExact;
}

TSPPDState::TSPPDState(TSPPDState&& other)
{
  pPickupDeliveryMap = other.pPickupDeliveryMap;
  pCostMatrix = other.pCostMatrix;
  pDomain = std::move(other.pDomain);
  pPath = std::move(other.pPath);
  pIsExact = other.pIsExact;

  other.pPickupDeliveryMap = nullptr;
  other.pCostMatrix = nullptr;
}

TSPPDState& TSPPDState::operator=(const TSPPDState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pPickupDeliveryMap = other.pPickupDeliveryMap;
  pCostMatrix = other.pCostMatrix;
  pDomain = other.pDomain;
  pPath = other.pPath;
  pIsExact = other.pIsExact;
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
  pDomain = std::move(other.pDomain);
  pPath = std::move(other.pPath);
  pIsExact = other.pIsExact;

  other.pPickupDeliveryMap = nullptr;
  other.pCostMatrix = nullptr;
  return *this;
}

bool TSPPDState::isEqual(const DPState* other) const noexcept
{
  // A state is equivalent to another if the set of next reachable nodes is the same
  //return pDomain == reinterpret_cast<const TSPPDState*>(other)->pDomain;
  return false;
}

bool TSPPDState::isStrictlyEqual(const DPState* other) const noexcept
{
  if (pDomain != reinterpret_cast<const TSPPDState*>(other)->pDomain)
  {
    return false;
  }

  auto otherState = reinterpret_cast<const TSPPDState*>(other);
  if ((pCost != otherState->pCost) || (pPath.back() != otherState->pPath.back()))
  {
    return false;
  }
  return true;
}

void TSPPDState::resetState() noexcept
{
  pCost = 0.0;

  // Initialize the set of nodes that can still
  // be visited from this state on.
  // Since this is the first/default state,
  // only pick-up nodes can be taken
  pDomain.clear();
  for (const auto& locIter : *pPickupDeliveryMap)
  {
    pDomain.insert(locIter.first);
  }

  // Clear cumulative path
  pPath.clear();

  // Set this state as default state
  this->setDefaultState();
}

DPState* TSPPDState::clone() const noexcept
{
  auto other = new TSPPDState(pPickupDeliveryMap, pCostMatrix);
  DPState::copyBaseDPState(other);
  other->pDomain = pDomain;
  return other;
}

void TSPPDState::updateState(const DPState* fromState, int64_t val)
{
  // Replace the state (override its internal data)
  auto fromStateCast = reinterpret_cast<const TSPPDState*>(fromState);

  pCost = fromStateCast->pCost;
  pCost += fromStateCast->pPath.empty() ?
          pCostMatrix->at(0).at(val) :
          pCostMatrix->at(fromStateCast->pPath.back()).at(val);

  pPath = fromStateCast->pPath;
  pPath.push_back(val);
  pDomain = fromStateCast->pDomain;

  // Remove the current value from the list of possible nodes that can be taken
  pDomain.erase(val);

  // Check if the current node is a pick-up node.
  // If so, add the correspondent delivery node
  if (pPickupDeliveryMap->find(val) != pPickupDeliveryMap->end())
  {
    pDomain.insert(pPickupDeliveryMap->at(val));
  }
}

double TSPPDState::getCostOnValue(int64_t value)
{
  return pPath.empty() ?
          pCostMatrix->at(0).at(value) :
          pCostMatrix->at(pPath.back()).at(value);
}

std::vector<DPState::UPtr> TSPPDState::nextStateList(int64_t lb, int64_t ub, double incumbent) const
{
  std::vector<DPState::UPtr> outStateList;

  // Instead of evaluating [lb, ub] values, evaluate only the values that can
  // be reached from this state
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
    // Keep track of the best values found so far
    auto costVal{pCost};
    costVal += pPath.empty() ?
            pCostMatrix->at(0).at(val) :
            pCostMatrix->at(pPath.back()).at(val);
    if (costVal >= incumbent)
    {
      // Skip states that lead to a cost higher than the incumbent,
      // i.e., apply pruning
      continue;
    }

    // Here there is a value that can lead to a valid next state.
    // The state is a clone of this state where "val" is applied to it.
    // Note: for this DP model, two states built with different values
    // will never be equivalent
    auto nextState = this->clone();
    nextState->updateState(this, val);
    outStateList.emplace_back(nextState);
  }

  return std::move(outStateList);
}

uint32_t TSPPDState::stateSelectForMerge(const std::vector<DPState::UPtr>& statesList) const
{
  assert(!statesList.empty());

  // Select the state with cumulative cost closer to this state
  uint32_t mergeIdx{0};
  double diff{std::numeric_limits<double>::max()};
  for (uint32_t idx{0}; idx < static_cast<uint32_t>(statesList.size()); ++idx)
  {
    const auto absVal = std::abs(this->pCost - statesList.at(idx)->cumulativeCost());
    if (absVal < diff)
    {
      diff = absVal;
      mergeIdx = idx;
    }
  }
  return mergeIdx;
}

void TSPPDState::mergeState(DPState* other) noexcept
{
  auto otherState = reinterpret_cast<const TSPPDState*>(other);
  if (otherState->pCost < pCost)
  {
    pCost = otherState->pCost;
    pPath = otherState->pPath;
  }

  for (auto node : otherState->pDomain)
  {
    pDomain.insert(node);
  }
}

std::string TSPPDState::toString() const noexcept
{
  std::stringstream ss;
  ss << "State ID: " << getUniqueId() << '\n';
  ss << "Cost: " << pCost << '\n';
  ss << "Is exact: " << (pIsExact ? "true" : "false") << '\n';
  ss << "Path: ";
  for (auto v : pPath)
  {
    ss << v << ", ";
  }
  ss << '\n';

  ss << "Domain: ";
  for (auto v : pDomain)
  {
    ss << v << ", ";
  }
  ss << '\n';
  return ss.str();
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

double TSPPD::calculateCost(const std::vector<int64_t>& path) const
{
  double cost{0.0};
  for (uint32_t idx{0}; idx < static_cast<int64_t>(path.size()) - 1; ++idx)
  {
    cost += pCostMatrix.at(path.at(idx)).at(path.at(idx+1));
  }
  return cost;
}

DPState::SPtr TSPPD::getInitialDPState() const noexcept
{
  return pInitialDPState;
}

DPState* TSPPD::getInitialDPStateRaw() noexcept
{
  return new TSPPDState(&pPickupDeliveryMap, &pCostMatrix, true);
}

void TSPPD::enforceConstraint(Arena* arena,
                              std::vector<std::vector<Node*>>& mddRepresentation,
                              std::vector<Node*>& newNodesList) const
{
}

};
