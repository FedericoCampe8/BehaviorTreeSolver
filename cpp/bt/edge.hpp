//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Behavior Tree edge.
//

#pragma once

#include <memory>  // for std::unique_ptr

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
   /// Note: this methods DO NOT modify the children of the current or given nodes
   void changeHead(Node* head);
   void changeTail(Node* tail);


   /// Returns the raw pointer to the head node.
   /// Note: use only if you know what you are doing!
   Node* getHead() const noexcept { return pHead; }

   /// Returns the raw pointer to the tail node.
   /// Note: use only if you know what you are doing!
   Node* getTail() const noexcept { return pTail; }

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

   /// Pointer to the domain on this edge
   cp::Variable::FiniteDomain* pDomain{nullptr};

   /// Boolean flag indicating whether or not this edge has been added to the nodes
   bool pEdgeAddedToNodes{false};

   /// Registers this edge on the head and tail nodes
   void setEdgeOnNodes();
};

}  // namespace btsolver
