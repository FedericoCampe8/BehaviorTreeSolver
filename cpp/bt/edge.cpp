#include "bt/edge.hpp"

#include <stdexcept>

namespace btsolver {

// Initialize unique Identifier for edges
uint32_t Edge::kNextID = 0;

Edge::Edge(Node* head, Node* tail)
: pEdgeId(Edge::kNextID++),
  pHead(head),
  pTail(tail)
{
  if (pHead == nullptr || pTail == nullptr)
  {
    throw std::invalid_argument("Edge - empty pointers to Head/Tail");
  }

  // Set this edge in the list of edges on the head/tail nodes
  setEdgeOnNodes();
}

Edge::~Edge()
{
  removeEdgeFromNodes();
}

void Edge::setEdgeOnNodes()
{
  if (pHead)
  {
    pHead->addOutgoingEdge(this->getUniqueId());
  }
  if (pTail)
  {
    pTail->addIncomingEdge(this->getUniqueId());
  }
  pEdgeAddedToNodes = true;
}

void Edge::removeEdgeFromNodes()
{
  if (pEdgeAddedToNodes)
  {
    if (pHead)
    {
      pHead->removeOutgoingEdge(this->getUniqueId());
    }

    if (pTail)
    {
      pTail->removeIncomingEdge(this->getUniqueId());
    }
    pEdgeAddedToNodes = false;
  }
}

void Edge::resetHead(Node* head)
{
  pHead = head;
}

void Edge::resetTail(Node* tail)
{
  pTail = tail;
}

}  // namespace btsolver
