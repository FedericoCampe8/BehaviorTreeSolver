#include "bt/edge.hpp"

#include <algorithm>  // for std::max, std::min
#include <stdexcept>  // for std::invalid_argument

#include "bt/opt_node.hpp"

namespace btsolver {

// Initialize unique Identifier for edges
uint32_t Edge::kNextID = 0;

Edge::Edge()
: pEdgeId(Edge::kNextID++)
{
}

Edge::Edge(Node* head, Node* tail)
: pEdgeId(Edge::kNextID++),
  pHead(head),
  pTail(tail)
{
  if (pHead == nullptr || pTail == nullptr)
  {
    throw std::invalid_argument("Edge - empty pointers to Head/Tail");
  }

  pHead->addOutgoingEdge(this);
  pTail->addIncomingEdge(this);
}

Edge::~Edge()
{
  removeEdgeFromNodes();
}

void Edge::removeEdgeFromNodes()
{
  removeHead();
  removeTail();
}

void Edge::removeHead() noexcept
{
  if (pHead)
  {
    pHead->removeOutgoingEdge(this);
  }
  pHead = nullptr;
}

void Edge::removeTail() noexcept
{
  if (pTail)
  {
    pTail->removeIncomingEdge(this);
  }
  pTail = nullptr;
}

void Edge::setHead(Node* node) noexcept
{
  if (node == nullptr || node == pHead || node == pTail)
  {
    return;
  }

  if (pHead)
  {
    removeHead();
  }
  pHead = node;
  pHead->addOutgoingEdge(this);
}

void Edge::setTail(Node* node) noexcept
{
  if (node == nullptr || node == pTail || node == pHead)
  {
    return;
  }

  if (pTail)
  {
    removeTail();
  }
  pTail = node;
  pTail->addIncomingEdge(this);
}

double Edge::getCostValue() const noexcept
{
  if (pTail == nullptr || pTail->getNodeType() != NodeType::State)
  {
    // The edge is not connected to anything, the cost is +INF
    return std::numeric_limits<double>::max();
  }

  auto state = pTail->cast<optimization::OptimizationState>();
  if (!isParallelEdge())
  {
    return state->getDPStateMutable()->cost(pDomainLowerBound);
  }
  else
  {
    double cost{0.0};
    auto dpState = state->getDPStateMutable();
    for (auto domVal{pDomainLowerBound}; domVal <= pDomainUpperBound; ++domVal)
    {
      if (isElementInDomain(domVal))
      {
        // Sum the cost only for the elements in the domain,
        // i.e., skip the holes
        cost += dpState->cost(domVal);
      }
    }
    return cost;
  }
}

std::pair<double, double> Edge::getCostBounds() const noexcept
{
  if (pTail == nullptr || pTail->getNodeType() != NodeType::State)
  {
    return {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
  }

  auto state = pTail->cast<optimization::OptimizationState>();
  if (!isParallelEdge())
  {
    auto cost = state->getDPStateMutable()->cost(pDomainLowerBound);
    return {cost, cost};
  }
  else
  {
    double lbCost{std::numeric_limits<double>::max()};
    double ubCost{std::numeric_limits<double>::lowest()};
    auto dpState = state->getDPStateMutable();
    for (auto domVal{pDomainLowerBound}; domVal <= pDomainUpperBound; ++domVal)
    {
      if (isElementInDomain(domVal))
      {
        const auto cost = dpState->cost(domVal);
        lbCost = std::min(lbCost, cost);
        ubCost = std::max(ubCost, cost);
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

uint32_t Edge::getDomainSize() const noexcept
{
  if (pDomainLowerBound > pDomainUpperBound)
  {
    return 0;
  }
  else if (pDomainLowerBound == pDomainUpperBound)
  {
    return 1;
  }
  else
  {
    return static_cast<uint32_t>(
            (pDomainUpperBound - pDomainLowerBound + 1) - pInvalidDomainElements.size());
  }
}

void Edge::finalizeDomain() noexcept
{
  if (pDomainLowerBound >= pDomainUpperBound)
  {
    return;
  }

  while(pDomainLowerBound <= pDomainUpperBound)
  {
    if (isElementInDomain(pDomainLowerBound))
    {
      break;
    }
    pDomainLowerBound++;
  }

  while(pDomainLowerBound <= pDomainUpperBound)
  {
    if (isElementInDomain(pDomainUpperBound))
    {
      break;
    }
    pDomainUpperBound--;
  }
}

void Edge::reinsertElementInDomain(int32_t element) noexcept
{
  if (element < pDomainLowerBound)
  {
    const auto oldLB = pDomainLowerBound;
    pDomainLowerBound = element;
    for (auto val{pDomainLowerBound+1}; val < oldLB; ++val)
    {
      pInvalidDomainElements.insert(val);
    }
  }
  else if (element > pDomainUpperBound)
  {
    const auto oldUB = pDomainUpperBound;
    pDomainUpperBound = element;
    for (auto val{oldUB+1}; val < pDomainUpperBound; ++val)
    {
      pInvalidDomainElements.insert(val);
    }
  }
  pInvalidDomainElements.erase(element);
}

void Edge::removeElementFromDomain(int32_t element) noexcept
{
  if (element == pDomainLowerBound)
  {
    pDomainLowerBound++;
  }
  else if (element == pDomainUpperBound)
  {
    pDomainUpperBound--;
  }
  else
  {
    // Create a "hole" in the domain
    pInvalidDomainElements.insert(element);
  }
}

bool Edge::isDomainEmpty() const noexcept
{
  // Check if all distinct elements in the domain
  // have been removed/marked as invalid elements
  if (pDomainLowerBound > pDomainUpperBound)
  {
    return true;
  }
  else if (static_cast<int32_t>(pInvalidDomainElements.size()) >
  (pDomainUpperBound - pDomainLowerBound + 1))
  {
    return true;
  }
  return false;
}

bool Edge::isElementInDomain(int32_t element) const noexcept
{
  if (pDomainLowerBound <= element && element <= pDomainUpperBound)
  {
    return pInvalidDomainElements.find(element) == pInvalidDomainElements.end();
  }
  return false;
}

}  // namespace btsolver
