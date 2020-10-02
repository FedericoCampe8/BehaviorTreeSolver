#include "mdd_optimization/node.hpp"

#include <cassert>
#include <stdexcept>  // for std::invalid_argument

namespace mdd {

// Initialize unique Identifier for nodes
uint32_t Node::kNextID = 0;

Node::Node(uint32_t layer, Variable* variable)
: pNodeId(Node::kNextID++),
  pLayer(layer),
  pVariable(variable),
  pIsDPStateChanged(false),
  pDefaultDPState(std::make_shared<DPState>())
{
  pDPState = pDefaultDPState;
}

Node::~Node()
{
  // Remove this node from all edges
  auto inEdgeListCopy = pInEdges;
  for (auto inEdge : inEdgeListCopy)
  {
    if (inEdge != nullptr)
    {
      inEdge->removeHead();
    }
  }

  auto outEdgeListCopy = pOutEdges;
  for (auto outEdge : outEdgeListCopy)
  {
    if (outEdge != nullptr)
    {
      outEdge->removeTail();
    }
  }
}

void Node::initializeNodeDomain()
{
  if (pVariable == nullptr)
  {
    throw std::runtime_error("Node - getValues: empty pointer to the variable");
  }

  pNodeDomain = pVariable->getAvailableValues();
}

void Node::addInEdge(Edge* edge)
{
  if (edge == nullptr)
  {
    throw std::invalid_argument("Node - add_in_edge: empty pointer to edge");
  }

  // Return if edge is already connected
  if (pInEdgeSet.find(edge->getUniqueId()) != pInEdgeSet.end())
  {
    return;
  }

  pInEdges.push_back(edge);
  pInEdgeSet.insert(edge->getUniqueId());

  // Set this node as the tail of the given edge
  edge->setHead(this);

  // Keep track of all incoming paths
  std::vector<EdgeList> newIncomingPaths;
  for (const auto& pathsTillHereIter : edge->getTail()->getIncomingPaths())
  {
    // For all paths getting to the tail, take the path and add the current edge
    for (int pathIdx{0}; pathIdx < pathsTillHereIter.second.size(); ++pathIdx)
    {
      newIncomingPaths.emplace_back(pathsTillHereIter.second[pathIdx]);
      newIncomingPaths.back().push_back(edge);
    }
  }

  // Consider the case where the tail is the root node
  // and there are no incoming paths.
  // The in-edge should still be added to the list of paths
  // arriving at this current node
  if (newIncomingPaths.empty())
  {
    newIncomingPaths.resize(1);
    newIncomingPaths.back().push_back(edge);
  }

  pIncomingPathsForEdge[edge->getUniqueId()] = newIncomingPaths;
}

void Node::addOutEdge(Edge* edge)
{
  if (edge == nullptr)
  {
    throw std::invalid_argument("Node - add_out_edge: empty pointer to edge");
  }

  // Return if edge is already connected
  if (pOutEdgeSet.find(edge->getUniqueId()) != pOutEdgeSet.end())
  {
    return;
  }

  pOutEdges.push_back(edge);
  pOutEdgeSet.insert(edge->getUniqueId());

  // Set this node as the head of the given edge
  edge->setTail(this);
}

void Node::removeInEdge(uint32_t position)
{
  auto edge = pInEdges.at(position);
  removeInEdgeGivenPtr(edge);
}

void Node::removeOutEdge(uint32_t position)
{
  auto edge = pOutEdges.at(position);
  removeOutEdgeGivenPtr(edge);
}

void Node::removeInEdgeGivenPtr(Edge* edge)
{
  if (edge == nullptr || (pInEdgeSet.find(edge->getUniqueId()) == pInEdgeSet.end()))
  {
    return;
  }

  auto iter = std::find(pInEdges.begin(), pInEdges.end(), edge);
  assert(iter != pInEdges.end());
  pInEdges.erase(iter);
  pInEdgeSet.erase(edge->getUniqueId());

  // Update the edge
  edge->removeHead();

  // Remove paths from this edge
  pIncomingPathsForEdge.erase(edge->getUniqueId());
}

void Node::removeOutEdgeGivenPtr(Edge* edge)
{
  if (edge == nullptr || (pOutEdgeSet.find(edge->getUniqueId()) == pOutEdgeSet.end()))
  {
    return;
  }

  auto iter = std::find(pOutEdges.begin(), pOutEdges.end(), edge);
  assert(iter != pOutEdges.end());
  pOutEdges.erase(iter);
  pOutEdgeSet.erase(edge->getUniqueId());

  // Update the edge
  edge->removeTail();
}

void Node::setSelectedEdge(Edge* edge)
{
  if (edge == nullptr)
  {
    throw std::invalid_argument("Node - set_selected_edge: empty pointer to edge");
  }
  pSelectedEdge = edge;
}

std::string Node::getNodeStringId() const noexcept
{
  if (pLayer == 0)
  {
    return "r";
  }
  if (pVariable && !pVariable->getName().empty())
  {
    return pVariable->getName() + "_" + std::to_string(pNodeId);
  }
  return "u" + std::to_string(pNodeId);
}

}  // namespace mdd
