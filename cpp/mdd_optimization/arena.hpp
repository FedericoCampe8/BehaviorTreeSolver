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
  using UPtr = std::unique_ptr<Arena>;
  using SPtr = std::shared_ptr<Arena>;

 public:
  Arena() = default;
  ~Arena() = default;

  Node* buildNode(uint32_t layer, Variable* variable = nullptr)
  {
    // TODO add garbage collector: check if size is greater than max and, if so,
    // resize the pool
    pNodePool.push_back(std::make_unique<Node>(layer, variable));
    pNodeArena[pNodePool.back()->getUniqueId()] = static_cast<uint32_t>((pNodePool.size() - 1));
    return pNodePool.back().get();
  }

  /// Create a new node edge returns its raw pointer
  Edge* buildEdge(Node* tail,
                  Node* head,
                  int64_t valueLB = std::numeric_limits<int64_t>::min(),
                  int64_t valueUB = std::numeric_limits<int64_t>::max())
  {
    // TODO add garbage collector: check if size is greater than max and, if so,
    // resize the pool
    pEdgePool.push_back(std::make_unique<Edge>(tail, head, valueLB, valueUB));
    pEdgeArena[pEdgePool.back()->getUniqueId()] = static_cast<uint32_t>((pEdgePool.size() - 1));
    return pEdgePool.back().get();
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
    pNodePool[pNodeArena.at(nodeId)].reset();
    pNodeArena.erase(nodeId);
  }

  /// Deletes the edge with given id
  void deleteEdge(uint32_t edgeId)
  {
    pEdgePool[pEdgeArena.at(edgeId)].reset();
    pEdgeArena.erase(edgeId);
  }

  /// Returns the pool of nodes
  const std::vector<Node::UPtr>& getNodePool() const noexcept { return pNodePool; }

  /// Returns the pool of edges
  const std::vector<Edge::UPtr>& getEdgePool() const noexcept { return pEdgePool; }

 private:
  /// Map from node id to its index in the node list
  using NodeArena = spp::sparse_hash_map<uint32_t, uint32_t>;

  /// Map from edge id to its index in the edge list
  using EdgeArena = spp::sparse_hash_map<uint32_t, uint32_t>;

 private:
   /// Node map
   NodeArena pNodeArena;

   /// Edge map
   EdgeArena pEdgeArena;

   /// List of all the node instances in the Behavior Tree
   std::vector<Node::UPtr> pNodePool;

   /// List of all the edge instances in the Behavior Tree
   std::vector<Edge::UPtr> pEdgePool;
};

}  // namespace mdd
