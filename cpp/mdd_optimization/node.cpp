#include "mdd_optimization/node.hpp"

#include <cassert>
#include <stdexcept>  // for std::invalid_argument

namespace mdd {

// Initialize unique Identifier for nodes
uint32_t Node::kNextID = 0;

Node::Node(uint32_t layer, Variable* variable)
: pNodeId(Node::kNextID++),
  pLayer(layer),
  pVariable(variable)
{
}

Node::~Node()
{
  // Remove this node from all edges
  auto inEdgeListCopy = pInEdges;
  for (auto inEdge : inEdgeListCopy)
  {
    if (inEdge != nullptr)
    {
      inEdge->removeTail();
    }
  }

  auto outEdgeListCopy = pOutEdges;
  for (auto outEdge : outEdgeListCopy)
  {
    if (outEdge != nullptr)
    {
      outEdge->removeHead();
    }
  }
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
  edge->setTail(this);
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
  edge->setHead(this);
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
  edge->removeTail();
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
  edge->removeHead();
}

void Node::setSelectedEdge(Edge* edge)
{
  if (edge == nullptr)
  {
    throw std::invalid_argument("Node - set_selected_edge: empty pointer to edge");
  }
  pSelectedEdge = edge;
}

}  // namespace mdd
