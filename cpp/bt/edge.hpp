//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Behavior Tree edge.
//

#pragma once

#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <utility>  // for std::pair

#include <sparsepp/spp.h>

#include "bt/node.hpp"
#include "cp/variable.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {

class SYS_EXPORT_CLASS Edge {
 public:
   using UPtr = std::unique_ptr<Edge>;
   using SPtr = std::shared_ptr<Edge>;
   using DomainPtr =std::shared_ptr<cp::Variable::FiniteDomain>;

 public:
   /// Constructs an edge from the head node to tail node.
   /// Throws std::invalid_argument on empty pointers
   Edge(Node* head, Node* tail);
   ~Edge();

   /// Returns this edge's unique identifier
   uint32_t getUniqueId() const noexcept { return pEdgeId; }

   /// Resets head and tail nodes
   void resetHead(Node* head = nullptr);
   void resetTail(Node* tail = nullptr);

   /// Change head and tail nodes.
   /// This method removes and adds this edge to from the old
   /// and to the new nodes respectively.
   /// However, this method DOES NOT remove or add any child.
   /// That needs to be done manually from the nodes.
   void changeHead(Node* head);
   void changeTail(Node* tail);

   /// Returns the raw pointer to the head node.
   /// Note: use only if you know what you are doing!
   Node* getHead() const noexcept { return pHead; }

   /// Returns the raw pointer to the tail node.
   /// Note: use only if you know what you are doing!
   Node* getTail() const noexcept { return pTail; }

   /// Returns the cost on the objective function on this edge.
   /// If this edge represents a parallel edge,
   /// the cost is the sum of all edges.
   /// If this edge is not connected to a state node, return +Inf
   double getCostValue() const noexcept;

   /// Returns the cost bounds on the objective function on this edge.
   /// Note: this makes sense only if the edge is a parallel edge.
   /// If not, use the faster version "getCostValue"
   std::pair<double, double> getCostBounds() const noexcept;

   /// Returns true if this edge represents a parallel edge.
   /// Returns false otherwise
   bool isParallelEdge() const noexcept { return pDomainLowerBound < pDomainUpperBound; }

   /// Sets the domain bounds for this edge.
   /// If "lowerBound" < "upperBound" this edge represents a parallel edge.
   /// Throws if "lowerBound" > "upperBound"
   void setDomainBounds(int32_t lowerBound, int32_t upperBound);

   /// Returns the original domain lower bound on this edge
   int32_t getDomainLowerBound() const noexcept { return pDomainLowerBound; }

   /// Returns the original domain lower bound on this edge
   int32_t getDomainUpperBound() const noexcept { return pDomainUpperBound; }

   /// Returns the size of the current domain
   uint32_t getDomainSize() const noexcept;

   /// Re-sets lower/upper bound w.r.t. the removed elements
   void finalizeDomain() noexcept;

   /// Re-inserts an element in the domain on this edge
   void reinsertElementInDomain(int32_t element) noexcept;

   /// Removes an element from the domain on this edge
   void removeElementFromDomain(int32_t element) noexcept;

   /// Returns whether or not the domain is empty
   bool isDomainEmpty() const noexcept;

   /// Returns whether or not the given element is part of the domain on this edge
   bool isElementInDomain(int32_t element) const noexcept;

   /// Sets the domain for this edge
   void setDomain(cp::Variable::FiniteDomain* domain) noexcept { pDomain = domain; }

   /// Sets the domain for this edge and make the edge owner of the domain.
   /// Note: this is usually done for temporary domains
   void setDomainAndOwn(cp::Variable::FiniteDomain* domain);

   /// Returns the pointer to the domain associated with this edge
   cp::Variable::FiniteDomain* getDomainMutable() const noexcept { return pDomain; }

   /// Removes this edge from the head and tail node
   void removeEdgeFromNodes();

 private:
   static uint32_t kNextID;

 private:
   /// Unique identifier for this edge used to retrieve
   /// this edge efficiently from the set of edges
   uint32_t pEdgeId{0};

   /// Pointer to the head node of this edge
   Node* pHead{nullptr};

   /// Pointer to the tail node of this edge
   Node* pTail{nullptr};

   /// Flag indicating whether this edge owns the domain or not
   bool pOwnsDomain{false};

   /// Lower bound on the represented domain
   int32_t pDomainLowerBound{std::numeric_limits<int32_t>::min()};

   /// Upper bound on the represented domain
   int32_t pDomainUpperBound{std::numeric_limits<int32_t>::max()};

   /// Sets of elements in [lb, ub] that are not part of the domain
   spp::sparse_hash_set<int32_t> pInvalidDomainElements;

   /// Pointer to the domain on this edge
   cp::Variable::FiniteDomain* pDomain{nullptr};

   /// Boolean flag indicating whether or not this edge has been added to the nodes
   bool pEdgeAddedToNodes{false};

   /// Registers this edge on the head and tail nodes
   void setEdgeOnNodes();
};

}  // namespace btsolver
