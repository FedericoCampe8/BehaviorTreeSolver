#include "bt/edge.hpp"

#include "bt/opt_node.hpp"

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

double Edge::getCostValue() const noexcept
{
  if (!pTail)
  {
    return std::numeric_limits<double>::max();
  }

  auto stateNode = reinterpret_cast<optimization::OptimizationState*>(pTail);
  if (!this->isParallelEdge())
  {
    return stateNode->getDPStateMutable()->cost(pDomainLowerBound);
  }
  else
  {
    double cost{0.0};
    auto dpState = stateNode->getDPStateMutable();
    for (auto domVal{pDomainLowerBound}; domVal <= pDomainUpperBound; ++domVal)
    {
      cost += dpState->cost(domVal);
    }
    return cost;
  }
}

std::pair<double, double> Edge::getCostBounds() const noexcept
{
  if (!pTail)
  {
    return {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
  }

  auto stateNode = reinterpret_cast<optimization::OptimizationState*>(pTail);
  if (!this->isParallelEdge())
  {
    auto cost = stateNode->getDPStateMutable()->cost(pDomainLowerBound);
    return {cost, cost};
  }
  else
  {
    double lbCost{std::numeric_limits<double>::max()};
    double ubCost{std::numeric_limits<double>::lowest()};
    auto dpState = stateNode->getDPStateMutable();
    for (auto domVal{pDomainLowerBound}; domVal <= pDomainUpperBound; ++domVal)
    {
      auto cost = dpState->cost(domVal);
      if (lbCost > cost)
      {
        lbCost = cost;
      }
      else if (ubCost < cost)
      {
        ubCost = cost;
      }
    }
    return {lbCost, ubCost};
  }
}

void Edge::setDomainBounds(int32_t lowerBound, int32_t upperBound)
{
  if (lowerBound > upperBound)
  {
    throw std::invalid_argument("Edge - setDomainBounds: lower bound "
            "greater than upper bound");
  }
  pDomainLowerBound = lowerBound;
  pDomainUpperBound = upperBound;
}

}  // namespace btsolver
