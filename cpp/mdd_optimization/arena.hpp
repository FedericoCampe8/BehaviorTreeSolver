//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for int32_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <vector>
#include <sparsepp/spp.h>

#include "mdd_optimization/edge.hpp"
#include "mdd_optimization/node.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS Arena {
 public:
  /// Map from node id to its index in the node list
  using NodeArena = spp::sparse_hash_map<uint32_t, Node*>;

  /// Map from edge id to its index in the edge list
  using EdgeArena = spp::sparse_hash_map<uint32_t, Edge*>;

  using UPtr = std::unique_ptr<Arena>;
  using SPtr = std::shared_ptr<Arena>;

 public:
  Arena() = default;
  ~Arena() = default;

  Node* buildNode(uint32_t layer, Variable* variable = nullptr)
  {
    auto newNode = new Node(layer, variable);
    pNodeArena[newNode->getUniqueId()] = newNode;
    return newNode;
  }

  /// Create a new node edge returns its raw pointer
  Edge* buildEdge(Node* tail,
                  Node* head,
                  int64_t valueLB = std::numeric_limits<int64_t>::min(),
                  int64_t valueUB = std::numeric_limits<int64_t>::max())
  {
    auto newEdge = new Edge(tail, head, valueLB, valueUB);
    pEdgeArena[newEdge->getUniqueId()] = newEdge;
    return newEdge;
  }

  /// Returns true if the arena contains the given node. Returns false otherwise
  bool containsNode(uint32_t nodeId) const noexcept
  {
    return pNodeArena.find(nodeId) != pNodeArena.end();
  }

  /// Returns true if the arena contains the given edge. Returns false otherwise
  bool containsEdge(uint32_t edgeId) const noexcept
  {
    return pEdgeArena.find(edgeId) != pEdgeArena.end();
  }

  /// Deletes the node with given id
  void deleteNode(uint32_t nodeId)
  {
    delete pNodeArena[nodeId];
    pNodeArena.erase(nodeId);
  }

  /// Deletes the edge with given id
  void deleteEdge(uint32_t edgeId)
  {
    delete pEdgeArena[edgeId];
    pEdgeArena.erase(edgeId);
  }

  /// Returns the pool of nodes
  const NodeArena& getNodePool() const noexcept { return pNodeArena; }

  /// Returns the pool of edges
  const EdgeArena& getEdgePool() const noexcept { return pEdgeArena; }

 private:
   /// Node map
   NodeArena pNodeArena;

   /// Edge map
   EdgeArena pEdgeArena;
};

}  // namespace mdd
