#include "mdd_optimization/edge.hpp"
#include "mdd_optimization/node.hpp"

#include <stdexcept>

namespace mdd {

// Initialize unique Identifier for edges
uint32_t Edge::kNextID = 0;

Edge::Edge(Node *tail, Node *head, int64_t valueLB, int64_t valueUB)
: pEdgeId(Edge::kNextID++),
  pHead(head),
  pTail(tail),
  pDomainLowerBound(valueLB),
  pDomainUpperBound(valueUB)
{
  if (pHead == nullptr)
  {
    throw std::invalid_argument("Edge - empty pointer to head node");
  }

  if (pTail == nullptr)
  {
    throw std::invalid_argument("Edge - empty pointer to tail node");
  }

  pHead->addInEdge(this);
  pTail->addOutEdge(this);
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
    pHead->removeInEdgeGivenPtr(this);
  }
  pHead = nullptr;
}

void Edge::removeTail() noexcept
{
  if (pTail)
  {
    pTail->removeOutEdgeGivenPtr(this);
  }
  pTail = nullptr;
}

void Edge::reverseEdge()
{
  auto oldHead = pHead;
  this->setHead(pTail);
  this->setTail(oldHead);
}

void Edge::setHead(Node* node)
{
  if (node == nullptr || node == pHead)
  {
    return;
  }

  if (pHead)
  {
    removeHead();
  }
  pHead = node;
  pHead->addInEdge(this);
}

void Edge::setTail(Node* node) noexcept
{
  if (node == nullptr || node == pTail)
  {
    return;
  }

  if (pTail)
  {
    removeTail();
  }
  pTail = node;
  pTail->addOutEdge(this);
}

void Edge::setDomainBounds(int64_t lowerBound, int64_t upperBound)
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

void Edge::reinsertElementInDomain(int64_t element) noexcept
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

void Edge::removeElementFromDomain(int64_t element) noexcept
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

bool Edge::isElementInDomain(int64_t element) const noexcept
{
  if (pDomainLowerBound <= element && element <= pDomainUpperBound)
  {
    return pInvalidDomainElements.find(element) == pInvalidDomainElements.end();
  }
  return false;
}

}  // namespace mdd
