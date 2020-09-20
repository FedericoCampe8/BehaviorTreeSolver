//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for int64_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr

#include <sparsepp/spp.h>

#include "system/system_export_defs.hpp"

// Forward declarations
namespace mdd {
class Node;
}  // namespace mdd

namespace mdd {

class SYS_EXPORT_CLASS Edge {
 public:
  using UPtr = std::unique_ptr<Edge>;

 public:
  /**
   * \brief Constructor, it does NOT take ownership of the given object instances.
   * \note given an edge, tail and head are, respectively:
   *       ------>
   *     tail   head
   */
  Edge(Node* tail,
       Node* head,
       int64_t valueLB = std::numeric_limits<int64_t>::min(),
       int64_t valueUB = std::numeric_limits<int64_t>::max());

  ~Edge();

  /// Returns this edge's unique identifier
  uint32_t getUniqueId() const noexcept { return pEdgeId; }

  /// Returns the head node of this edge
  Node* getHead() const noexcept { return pHead; }

  /// Returns the tail node of this edge
  Node* getTail() const noexcept { return pTail; }

  /// Returns the value on this edge
  int64_t getValue() const noexcept { return pDomainLowerBound; }

  /// Sets the head of this edge
  void setHead(Node *node);

  /// Sets the tail of this edge
  void setTail(Node* node) noexcept;

  /// Remove the head of this node and set it to nullptr
  void removeHead() noexcept;

  /// Remove the tail of this node and set it to nullptr
  void removeTail() noexcept;

  /// Removes this edge from the head and tail node
  void removeEdgeFromNodes();

  /// Sets the domain bounds for this edge.
  /// If "lowerBound" < "upperBound" this edge represents a parallel edge.
  /// Throws if "lowerBound" > "upperBound"
  void setDomainBounds(int64_t lowerBound, int64_t upperBound);

  /// Returns the original domain lower bound on this edge
  int64_t getDomainLowerBound() const noexcept { return pDomainLowerBound; }

  /// Returns the original domain lower bound on this edge
  int64_t getDomainUpperBound() const noexcept { return pDomainUpperBound; }

  /// Returns the size of the current domain
  uint32_t getDomainSize() const noexcept;

  /// Re-inserts an element in the domain on this edge.
  /// Note: this can expand the current domain
  void reinsertElementInDomain(int64_t element) noexcept;

  /// Removes an element from the domain on this edge
  void removeElementFromDomain(int64_t element) noexcept;

  /// Returns whether or not the domain is empty
  bool isDomainEmpty() const noexcept;

  /// Returns whether or not the given element is part of the domain on this edge
  bool isElementInDomain(int64_t element) const noexcept;

 private:
   static uint32_t kNextID;

 private:
   /// Unique identifier for this edge used to retrieve
   /// this edge efficiently from the set of edges
   uint32_t pEdgeId{0};

   /// Head node on this edge
   Node* pHead{nullptr};

   /// Tail node on this edge
   Node* pTail{nullptr};

   /// Lower bound on the represented domain
   int64_t pDomainLowerBound{std::numeric_limits<int64_t>::min()};

   /// Upper bound on the represented domain
   int64_t pDomainUpperBound{std::numeric_limits<int64_t>::max()};

   /// Sets of elements in [lb, ub] that are not part of the domain
   spp::sparse_hash_set<int32_t> pInvalidDomainElements;
};

}  // namespace mdd
