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

  if (pOwnsDomain)
  {
    // Delete the domain if owned by this edge
    delete pDomain;
  }
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

void Edge::changeHead(Node* head)
{
  if (head == pHead)
  {
    return;
  }

  // Remove this edge from the previous node
  if (pHead)
  {
    pHead->removeOutgoingEdge(this->getUniqueId());
  }

  // Set this edge on the new node
  pHead = head;
  if (pHead)
  {
    pHead->addOutgoingEdge(this->getUniqueId());
  }
}

void Edge::changeTail(Node* tail)
{
  if (tail == pTail)
  {
    return;
  }

  // Remove this edge from the previous node
  if (pTail)
  {
    pTail->removeIncomingEdge(this->getUniqueId());
  }

  // Set this edge on the new node
  pTail = tail;
  if (pTail)
  {
    pTail->addIncomingEdge(this->getUniqueId());
  }
}

void Edge::setDomainAndOwn(cp::Variable::FiniteDomain* domain)
{
  pDomain = domain;
  pOwnsDomain = true;
}

}  // namespace btsolver
