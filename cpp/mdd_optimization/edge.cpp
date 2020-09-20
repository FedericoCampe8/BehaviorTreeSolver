#include "mdd_optimization/edge.hpp"
#include "mdd_optimization/node.hpp"

#include <stdexcept>

namespace mdd {

Edge::Edge(Node *tail, Node *head, int64_t value)
: pHead(head),
  pTail(tail),
  pValue(value)
{
  if (pHead == nullptr)
  {
    throw std::invalid_argument("Edge - empty pointer to head node");
  }

  if (pTail == nullptr)
  {
    throw std::invalid_argument("Edge - empty pointer to tail node");
  }
}

void Edge::setHead(Node* node)
{
  if (node == nullptr)
  {
    throw std::invalid_argument("Edge - setHead: empty pointer to head node");
  }

  // TODO when the head is replaced, change outgoing edges on the node as well
  pHead = node;
}

}  // namespace mdd
