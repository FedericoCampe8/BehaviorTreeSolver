//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for int64_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr

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
   */
  Edge(Node *tail, Node *head, int64_t value);

  /// Returns the head node of this edge
  Node* getHead() const noexcept {
    return pHead;
  }

  /// Returns the tail node of this edge
  Node* getTail() const noexcept {
    return pTail;
  }

  /// Returns the value on this edge
  int64_t getValue() const noexcept {
    return pValue;
  }

  /// Sets the head on this edge
  void setHead(Node *node);

private:
    Node* pHead{nullptr};
    Node* pTail{nullptr};
    int64_t pValue{std::numeric_limits<int64_t>::min()};
};

}  // namespace mdd
