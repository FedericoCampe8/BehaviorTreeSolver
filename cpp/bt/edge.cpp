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
}

}  // namespace btsolver
