#include "mdd_optimization/node.hpp"

#include <stdexcept>

namespace mdd {

Node::Node(Variable* variable, uint32_t layer)
: pLayer(layer),
  pVariable(variable)
{
  if (pVariable == nullptr)
  {
    throw std::invalid_argument("Node - empty pointer to variable");
  }
}

void Node::addInEdge(Edge* edge)
{
  if (edge == nullptr)
  {
    throw std::invalid_argument("Node - add_in_edge: empty pointer to edge");
  }
  pInEdges.push_back(edge);
}

void Node::addOutEdge(Edge* edge)
{
  if (edge == nullptr)
  {
    throw std::invalid_argument("Node - add_out_edge: empty pointer to edge");
  }
  pOutEdges.push_back(edge);
}

void Node::removeInEdge(uint32_t position)
{
  pInEdges.erase(pInEdges.begin()+position);
}

void Node::removeOutEdge(uint32_t position)
{
  pOutEdges.erase(pOutEdges.begin()+position);
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
