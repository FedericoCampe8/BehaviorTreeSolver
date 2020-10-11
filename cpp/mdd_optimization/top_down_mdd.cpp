#include "mdd_optimization/top_down_mdd.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

namespace mdd {

MDDTDEdge::MDDTDEdge(int32_t tailLayer, int32_t tailIdx, int32_t headIdx)
: layer(tailLayer),
  tail(tailIdx),
  head(headIdx)
{
}

TopDownMDD::TopDownMDD(MDDProblem::SPtr problem, uint32_t width)
: pProblem(problem),
  pMaxWidth(width)
{
  if (pProblem == nullptr)
  {
    throw std::invalid_argument("TopDownMDD - empty pointer to the problem instance");
  }

  // Set the number of layers of this MDD which corresponds to the
  // number of variables
  pNumLayers = static_cast<uint32_t>(getVariablesList().size());
  for (uint32_t lidx{0}; lidx < pNumLayers+1; ++lidx)
  {
    pStartDefaultStateIdxOnLevel[lidx] = 0;
  }

  // Allocate all edges in the MDD
  allocateEdges();

  // Allocate all states: one state for each node
  if (getConstraintsList().size() != 1)
  {
    throw std::invalid_argument("TopDownMDD - invalid model: invalid number of constraints");
  }
  allocateStates(getConstraintsList().at(0));
}

void TopDownMDD::allocateEdges()
{
  pLayerEdgeList.reserve(pNumLayers);
  for (uint32_t lidx{0}; lidx < pNumLayers; ++lidx)
  {
    // Compute the max number of edges on each layer
    uint64_t maxNumEdges{pMaxWidth * pMaxWidth};
    if (lidx == 0)
    {
      // First layer has only one node
      maxNumEdges = pMaxWidth;
    }

    EdgeList edgeList;
    edgeList.reserve(maxNumEdges);
    for (uint64_t eidx{0}; eidx < maxNumEdges; ++eidx)
    {

      int32_t tailIdx{1};
      if (lidx == 0)
      {
        // All edges in the first layer have the same tail, i.e., the root)
        tailIdx = 0;
      }
      else
      {
        tailIdx = (eidx == 0 ? 0 : eidx / pMaxWidth);
      }

      int32_t headIdx = static_cast<int32_t>(eidx % pMaxWidth);
      if (lidx == (pNumLayers - 1))
      {
        // All heads on the last layer point to the same node, i.e., the tail
        headIdx = 0;
      }

      edgeList.push_back(std::make_unique<MDDTDEdge>(lidx, tailIdx, headIdx));
    }
    pLayerEdgeList.push_back(std::move(edgeList));
  }
}

void TopDownMDD::allocateStates(MDDConstraint::SPtr con)
{
  if (con == nullptr)
  {
    throw std::invalid_argument("TopDownMDD - allocateStates: empty pointer to the constraint");
  }

  // Reserve the memory for the layers plus the root
  pMDDStateMatrix.reserve(pNumLayers + 1);

  // Add the root state
  StateList rootStateList;
  pMDDStateMatrix.push_back(std::move(rootStateList));
  pMDDStateMatrix.front().push_back(DPState::UPtr(con->getInitialDPStateRaw()));

  // Add replacement on root state
  std::vector<DPState::UPtr> rootRepStateList;
  pReplacedStatesMatrix[0] = std::move(rootRepStateList);
  for (uint32_t lidx{0}; lidx < pNumLayers - 1; ++lidx)
  {
    StateList stateList;
    stateList.reserve(pMaxWidth);

    for (uint32_t widx{0}; widx < pMaxWidth; ++widx)
    {
      stateList.push_back(DPState::UPtr(con->getInitialDPStateRaw()));
    }

    // Add state for current layer
    pMDDStateMatrix.push_back(std::move(stateList));

    // Add replacement states for current layer
    std::vector<DPState::UPtr> repStateList;
    pReplacedStatesMatrix[lidx+1] = std::move(repStateList);
  }

  // Add tail node
  StateList tailStateList;
  pMDDStateMatrix.push_back(std::move(tailStateList));
  pMDDStateMatrix.back().push_back(DPState::UPtr(con->getInitialDPStateRaw()));

  // Add replacement state on tail node
  std::vector<DPState::UPtr> tailRepStateList;
  pReplacedStatesMatrix[pNumLayers] = std::move(tailRepStateList);
}

bool TopDownMDD::isLeafState(uint32_t layerIdx, uint32_t nodeIdx) const
{
  bool hasOutgoingEdges{false};
  for (const auto& outEdge : pLayerEdgeList.at(layerIdx))
  {
    if (outEdge->tail == nodeIdx && outEdge->isActive)
    {
      hasOutgoingEdges = true;
      break;
    }
  }

  if (!hasOutgoingEdges)
  {
    // The node doesn't have outgoing edges, check if it can be reached
    if (layerIdx == 0)
    {
      // Root nodes cannot have incoming edges, return asap
      return true;
    }
    else
    {
      // Check if the edge can be reached
      for (const auto& inEdge : pLayerEdgeList.at(layerIdx - 1))
      {
        if (inEdge->head == nodeIdx && inEdge->isActive)
        {
          // Edge can be reached and it doesn't have outgoing edges, return asap
          return true;
        }
      }
    }
  }

  return false;
}

void TopDownMDD::storeLeafNodes(double incumbent)
{
  for (uint32_t lidx{1}; lidx < static_cast<uint32_t>(pMDDStateMatrix.size() - 1); ++lidx)
  {
    for (uint32_t sidx{0}; sidx < pMDDStateMatrix.at(lidx).size(); ++sidx)
    {
      if (isLeafState(lidx, sidx))
      {
        // Copy the state and store it into the queue
        auto clonedState =pMDDStateMatrix.at(lidx).at(sidx)->clone();
        assert(clonedState != nullptr);

        // Store the cloned state into the list of replacement states
        // if the cost of the cloned state is less than the incumbent
        if (clonedState->cumulativeCost() < incumbent)
        {
          pReplacedStatesMatrix.at(lidx).push_back(DPState::UPtr(clonedState));
        }
      }
    }
  }
}

bool TopDownMDD::hasStoredStates() const noexcept
{
  for (auto& it : pReplacedStatesMatrix)
  {
    if (!it.second.empty())
    {
      return true;
    }
  }
  return false;
}

void TopDownMDD::rebuildMDDFromStoredStates()
{
  // Get the state to use to re-build the MDD
  auto state = getStateFromHistory();
  assert(state != nullptr);

  const auto& path = state->cumulativePath();

  // The first node is the root node
  auto tailNode = getNodeState(0, 0);
  for (int layerIdx{0}; layerIdx < static_cast<int>(path.size()); ++layerIdx)
  {
    // Each value represents an arc directed from the previous node
    // to the following node using the left path in the MDD
    pLayerEdgeList.at(layerIdx).at(0)->isActive = true;
    pLayerEdgeList.at(layerIdx).at(0)->value = path.at(layerIdx);

    auto nextNode = getNodeState(layerIdx + 1, 0);
    nextNode->updateState(tailNode, path.at(layerIdx));
    nextNode->setNonDefaultState();
    pStartDefaultStateIdxOnLevel[layerIdx + 1]++;

    // Update the pointer to the tail node
    tailNode = nextNode;
  }
}

DPState::UPtr TopDownMDD::getStateFromHistory()
{
  // Start a counter to avoid looping forever
  uint32_t totLayersCtr{0};

  // Consider next layer
  pHistoryStateLayerPtr = (pHistoryStateLayerPtr + 1) % (pNumLayers + 1);
  pHistoryStateLayerPtr = (pHistoryStateLayerPtr == 0) ? 1 : pHistoryStateLayerPtr;
  while (totLayersCtr++ < static_cast<uint32_t>(pReplacedStatesMatrix.size()))
  {
    auto& nodesQueueList = pReplacedStatesMatrix[pHistoryStateLayerPtr];
    if (!nodesQueueList.empty())
    {
      // Get and return the best node in the list
      auto bestIt = nodesQueueList.begin();
      double bestCost{std::numeric_limits<double>::max()};
      for (auto it = nodesQueueList.begin(); it != nodesQueueList.end(); ++it)
      {
        if ((*it)->cumulativeCost() < bestCost)
        {
          bestCost = (*it)->cumulativeCost();
          bestIt = it;
        }
      }

      // Remove and return the best state
      auto stateToReturn = std::move(*bestIt);
      nodesQueueList.erase(bestIt);

      // Return the pointer to the state
      return std::move(stateToReturn);
    }

    // Go to next layer
    pHistoryStateLayerPtr = (pHistoryStateLayerPtr + 1) % (pNumLayers + 1);
    pHistoryStateLayerPtr = (pHistoryStateLayerPtr == 0) ? 1 : pHistoryStateLayerPtr;
  }

  return nullptr;
}

void TopDownMDD::resetGraph(bool resetStatesQueue)
{
  // Deactivate all edges
  for (const auto& edgeList : pLayerEdgeList)
  {
    for (auto& edge : edgeList)
    {
      edge->isActive = false;
    }
  }

  // Reset all states
  for (const auto& stateList : pMDDStateMatrix)
  {
    for (auto& state : stateList)
    {
      // Reset the state and make sure it is set as a default state
      state->resetState();
      state->setNonDefaultState(true);
    }
  }

  for (auto& it : pStartDefaultStateIdxOnLevel)
  {
    it.second = 0;
  }

  if (resetStatesQueue)
  {
    for (auto& it : pReplacedStatesMatrix)
    {
      it.second.clear();
    }
  }
}

uint32_t TopDownMDD::getIndexOfFirstDefaultStateOnLayer(uint32_t layerIdx) const
{
  return pStartDefaultStateIdxOnLevel.at(layerIdx);

  // The following also works but it might be slightly slower for
  // a large number of states
  /*
  uint32_t idx{0};
  for (const auto& state : pMDDStateMatrix.at(layerIdx))
  {
    if (state->isDefaultState())
    {
      return idx;
    }
    ++idx;
  }
  return pMaxWidth;
  */
}

void TopDownMDD::replaceState(uint32_t layerIdx, uint32_t nodeIdx, DPState* currState, int64_t val,
                              bool storeDiscardedStates, double incumbent)
{
  assert(currState != nullptr);

  auto stateToReplace = getNodeState(layerIdx, nodeIdx);
  if(stateToReplace->isDefaultState())
  {
    // Default states should be replaced one at a time from left to right
    assert(nodeIdx == pStartDefaultStateIdxOnLevel[layerIdx]);
    pStartDefaultStateIdxOnLevel[layerIdx]++;
  }

  // Update the state and set it as non-default
  if (storeDiscardedStates)
  {
    // Copy the state and store it into the queue
    auto clonedState = stateToReplace->clone();
    assert(clonedState != nullptr);

    // Store the cloned state into the list of replacement states
    // if the cost of the cloned state is less than the incumbent
    if (clonedState->cumulativeCost() < incumbent)
    {
      pReplacedStatesMatrix.at(layerIdx).push_back(DPState::UPtr(clonedState));
    }
  }

  // Replace the state
  stateToReplace->updateState(currState, val);
  stateToReplace->setNonDefaultState();
}

std::vector<MDDTDEdge*> TopDownMDD::getActiveEdgesOnLayer(uint32_t layerIdx) const
{
  std::vector<MDDTDEdge*> edgeList;
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->isActive)
    {
      edgeList.push_back(edge.get());
    }
  }
  return edgeList;
}

MDDTDEdge* TopDownMDD::getEdgeOnHeadMutable(uint32_t layerIdx, uint32_t headIdx) const
{
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->head == headIdx && edge->isActive)
    {
      return edge.get();
    }
  }
  return nullptr;
}

void TopDownMDD::disableEdge(uint32_t layerIdx,  uint32_t tailIdx, uint32_t headIdx)
{
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->tail == tailIdx && edge->head == headIdx)
    {
      edge->isActive = false;
      return;
    }
  }
  assert(false);
}

void TopDownMDD::enableEdge(uint32_t layerIdx,  uint32_t tailIdx, uint32_t headIdx)
{
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->tail == tailIdx && edge->head == headIdx)
    {
      edge->isActive = true;
      return;
    }
  }
  assert(false);
}

void TopDownMDD::setEdgeValue(uint32_t layerIdx,  uint32_t tailIdx, uint32_t headIdx, int64_t val)
{
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->tail == tailIdx && edge->head == headIdx)
    {
      edge->value = val;
      return;
    }
  }
  assert(false);
}

bool TopDownMDD::isReachable(uint32_t layerIdx, uint32_t headIdx) const
{
  if (layerIdx == 0)
  {
    // First node, i.e., root node, is always reachable
    return true;
  }

  // Check for at least one active node pointing to the given head
  for (const auto& edge : pLayerEdgeList.at(layerIdx-1))
  {
    if (edge->head == headIdx && edge->isActive)
    {
      return true;
    }
  }

  return false;
}

void TopDownMDD::removeState(uint32_t layerIdx, uint32_t headIdx)
{
  (pMDDStateMatrix.at(layerIdx))[headIdx].reset();
}

void TopDownMDD::printMDD(const std::string& outFileName) const
{
  std::string ppMDD = "digraph D {\n";
  for (uint32_t lidx{0}; lidx < pNumLayers; ++lidx)
  {
    for (auto& edge : pLayerEdgeList.at(lidx))
    {
      if (!edge->isActive)
      {
        continue;
      }

      std::string tailPrefix{"U_"};
      std::string headPrefix{"U_"};
      if (lidx == 0)
      {
        tailPrefix = "R_";
      }
      tailPrefix += std::to_string(edge->layer) + "_" + std::to_string(edge->tail);

      if (lidx == (pNumLayers-1))
      {
        headPrefix = "T_";
      }
      headPrefix += std::to_string(edge->layer + 1) + "_" + std::to_string(edge->head);

      std::string newEdge = tailPrefix + " -> " + headPrefix;
      newEdge += std::string("[label=") + "\"" + std::to_string(edge->value)  +  "\"]\n";
      ppMDD += "\t" + newEdge;
    }
  }
  ppMDD += "}";

 std::ofstream outFile;
 outFile.open(outFileName + ".dot");
 outFile << ppMDD;
 outFile.close();
}

}  // namespace mdd
