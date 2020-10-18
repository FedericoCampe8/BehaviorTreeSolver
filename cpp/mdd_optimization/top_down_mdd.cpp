#include "mdd_optimization/top_down_mdd.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

namespace mdd {

MDDTDEdge::MDDTDEdge()
: isActive(false),
  layer(-1),
  tail(-1),
  head(-1)
{
  valuesList.push_back(std::numeric_limits<int64_t>::max());
  costList.push_back(std::numeric_limits<double>::max());
}

MDDTDEdge::MDDTDEdge(int32_t tailLayer, int32_t tailIdx, int32_t headIdx)
: isActive(false),
  layer(tailLayer),
  tail(tailIdx),
  head(headIdx)
{
  valuesList.push_back(std::numeric_limits<int64_t>::max());
  costList.push_back(std::numeric_limits<double>::max());
}

std::string MDDTDEdge::toString() const noexcept
{
  std::string info{""};
  info += "edge (" + std::to_string(layer) + "): " +
          std::to_string(tail) + " -> " + std::to_string(head) + '\n';
  info += isActive ? "active\n" : "not active\n";
  info += "Path: ";
  for (auto val : valuesList)
  {
    info += std::to_string(val) + ", ";
  }
  info.pop_back();
  info.pop_back();
  return info;
}

TopDownMDD::TopDownMDD(MDDProblem::SPtr problem, uint32_t width)
: pProblem(problem),
  pMaxWidth(width)
{
  if (pProblem == nullptr)
  {
    throw std::invalid_argument("TopDownMDD - empty pointer to the problem instance");
  }

  if (getConstraintsList().size() != 1)
  {
    throw std::invalid_argument("TopDownMDD - invalid model: invalid number of constraints");
  }

  // Set the number of layers of this MDD which corresponds to the
  // number of variables.
  // Each layer has a set of edges and a set of nodes.
  // Edges fully connect the nodes on that layer with the nodes
  // on the next layer starting from the root node
  pNumLayers = static_cast<uint32_t>(getVariablesList().size());

  // Allocate all edges in the MDD
  allocateEdges();

  // Allocate all states: one state for each node
  allocateStates(getConstraintsList().at(0));
}

void TopDownMDD::allocateEdges()
{
  pLayerEdgeList.reserve(pNumLayers);
  for (uint32_t lidx{0}; lidx < pNumLayers; ++lidx)
  {
    // Compute the max number of edges on each layer.
    // This corresponds to width^2 since layers are fully connected
    uint64_t maxNumEdges{pMaxWidth * pMaxWidth};
    if (lidx == 0 || (lidx == pNumLayers - 1))
    {
      // First and last layer have only one node
      maxNumEdges = pMaxWidth;
    }

    // For each layer, prepare a list of edges
    EdgeList edgeList;
    edgeList.reserve(maxNumEdges);
    for (uint64_t eidx{0}; eidx < maxNumEdges; ++eidx)
    {

      int32_t tailIdx{1};
      int32_t headIdx = static_cast<int32_t>(eidx % pMaxWidth);
      if (lidx == 0)
      {
        // All edges in the first layer have the same tail, i.e., the root
        tailIdx = 0;
      }
      else
      {
        tailIdx = (eidx == 0 ? 0 : eidx / pMaxWidth);
      }

      if (lidx == pNumLayers - 1)
      {
           // All edges in the last layer have the same head, i.e., the tail node
           headIdx = 0;
           tailIdx = eidx;
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

  // The number of nodes is equal to the number of layers plus one (the tail node)
  pMDDStateMatrix.reserve(pNumLayers + 1);

  // Add the (only) root state
  StateList rootStateList;
  pMDDStateMatrix.push_back(std::move(rootStateList));
  pMDDStateMatrix.front().push_back(DPState::UPtr(con->getInitialDPStateRaw()));
  for (uint32_t lidx{1}; lidx < pNumLayers; ++lidx)
  {
    StateList stateList;
    stateList.reserve(pMaxWidth);
    for (uint32_t widx{0}; widx < pMaxWidth; ++widx)
    {
      stateList.push_back(DPState::UPtr(con->getInitialDPStateRaw()));
    }

    // Add state for current layer
    pMDDStateMatrix.push_back(std::move(stateList));
  }

  // Add (only) tail node
  StateList tailStateList;
  pMDDStateMatrix.push_back(std::move(tailStateList));
  pMDDStateMatrix.back().push_back(DPState::UPtr(con->getInitialDPStateRaw()));
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

  if (hasOutgoingEdges)
  {
    // If the node has at least one outgoing edge,
    // it means the node it is not a leaf state, return asap
    return false;
  }

  // The node doesn't have outgoing edges, check if it can be reached
  if (layerIdx == 0)
  {
    // Root nodes cannot have incoming edges, return asap
    return true;
  }
  else
  {
    // Check if the edge can be reached by checking all incoming edges
    // and verifying that there is at least one active edge into the node
    for (const auto& inEdge : pLayerEdgeList.at(layerIdx - 1))
    {
      if (inEdge->head == nodeIdx && inEdge->isActive)
      {
        // Edge can be reached and it doesn't have outgoing edges, return asap
        return true;
      }
    }
  }

  return false;
}

void TopDownMDD::rebuildMDDFromState(DPState::UPtr state)
{
  assert(state != nullptr);

  // Get the path up to the state
  const auto& path = state->cumulativePath();

  // The first node is the root node
  auto tailNode = getNodeState(0, 0);
  for (int layerIdx{0}; layerIdx < static_cast<int>(path.size()); ++layerIdx)
  {
    // Each value represents an arc directed from the previous node
    // to the following node using the left path in the MDD
    pLayerEdgeList.at(layerIdx).at(0)->isActive = true;
    pLayerEdgeList.at(layerIdx).at(0)->valuesList[0] = path.at(layerIdx);

    auto nextNode = getNodeState(layerIdx + 1, 0);
    nextNode->updateState(tailNode, path.at(layerIdx));
    nextNode->setNonDefaultState();

    // Update the pointer to the tail node
    tailNode = nextNode;
  }
}

void TopDownMDD::resetGraph()
{
  // Deactivate all edges
  for (const auto& edgeList : pLayerEdgeList)
  {
    for (auto& edge : edgeList)
    {
      edge->isActive = false;
      edge->valuesList.clear();
      edge->valuesList.push_back(std::numeric_limits<int64_t>::max());
      edge->costList.clear();
      edge->costList.push_back(std::numeric_limits<double>::max());
    }
  }

  // Reset all states
  for (const auto& stateList : pMDDStateMatrix)
  {
    for (auto& state : stateList)
    {
      // Reset the state and make sure it is set as a default state
      state->resetState();
      state->setDefaultState();
      state->setExact(true);
    }
  }
}

MDDTDEdge* TopDownMDD::getEdgeMutable(uint32_t layerIdx,  uint32_t tailIdx, uint32_t headIdx) const
{
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->tail == tailIdx && edge->head == headIdx)
    {
      return edge.get();
    }
  }
  return nullptr;
}

uint32_t TopDownMDD::getIndexOfFirstDefaultStateOnLayer(uint32_t layerIdx) const
{
  uint32_t idx{0};
  for (const auto& state : pMDDStateMatrix.at(layerIdx))
  {
    if (state->isDefaultState())
    {
      return idx;
    }
    idx++;
  }
  return pMaxWidth;
  // return pStartDefaultStateIdxOnLevel.at(layerIdx);
}

DPState::UPtr TopDownMDD::replaceState(uint32_t layerIdx, uint32_t nodeIdx, DPState::UPtr state)
{
  auto stateToReplace = getNodeState(layerIdx, nodeIdx);

  // Keep track of the state to replace
  auto oldState = std::move(pMDDStateMatrix[layerIdx][nodeIdx]);
  pMDDStateMatrix[layerIdx][nodeIdx] = std::move(state);
  pMDDStateMatrix[layerIdx][nodeIdx]->setNonDefaultState();
  return std::move(oldState);
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

std::vector<MDDTDEdge*> TopDownMDD::getActiveEdgesOnLayerGivenTail(uint32_t layerIdx,
                                                                   uint32_t tailIdx) const
{
  std::vector<MDDTDEdge*> edgeList;
  uint32_t start{tailIdx * pMaxWidth};
  for (uint32_t idx{0}; idx < pMaxWidth; ++idx)
  {
    if (layerIdx == pNumLayers - 1)
    {
      start = 0;
      if (tailIdx != idx)
      {
        continue;
      }
    }
    const auto& edge = pLayerEdgeList.at(layerIdx).at(start + idx);
    if (edge->isActive)
    {
      edgeList.push_back(edge.get());
    }
  }
  return edgeList;
}

std::vector<MDDTDEdge*> TopDownMDD::getEdgeOnHeadMutable(uint32_t layerIdx, uint32_t headIdx) const
{
  std::vector<MDDTDEdge*> edgeList;
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->head == headIdx && edge->isActive)
    {
      edgeList.push_back(edge.get());
    }
  }
  return edgeList;
}

void TopDownMDD::disableEdge(uint32_t layerIdx, uint32_t tailIdx, uint32_t headIdx)
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

void TopDownMDD::enableEdge(uint32_t layerIdx, uint32_t tailIdx, uint32_t headIdx)
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

void TopDownMDD::setEdgeValue(uint32_t layerIdx, uint32_t tailIdx, uint32_t headIdx, int64_t val)
{
  for (const auto& edge : pLayerEdgeList.at(layerIdx))
  {
    if (edge->tail == tailIdx && edge->head == headIdx)
    {
      edge->valuesList[0] = val;
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

      // Create edge label
      std::string edgeLabel{""};
      for (auto val : edge->valuesList)
      {
        edgeLabel += std::to_string(val) + ", ";
      }
      edgeLabel.pop_back();
      edgeLabel.pop_back();

      newEdge += std::string("[label=") + "\"" + edgeLabel +  "\"]\n";
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
