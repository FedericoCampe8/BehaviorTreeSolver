//
// Copyright OptiLab 2020. All rights reserved.
//
// Area of memory containing the pointers to nodes and edges
// of the Behavior Tree.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::unique_ptr
#include <string>
#include <vector>

#include <sparsepp/spp.h>

#include "bt/blackboard.hpp"
#include "bt/node.hpp"
#include "bt/edge.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {

class SYS_EXPORT_CLASS BehaviorTreeArena {
 public:
  using UPtr = std::unique_ptr<BehaviorTreeArena>;
  using SPtr = std::shared_ptr<BehaviorTreeArena>;

 public:
  BehaviorTreeArena();
  ~BehaviorTreeArena() = default;

  /// Create a new node and returns its raw pointer
  template<typename NodeType>
  Node* buildNode(const std::string& name)
  {
    // TODO add garbage collector: check if size is greater than max and, if so,
    // resize the pool
    pNodePool.push_back(std::make_unique<NodeType>(name, this, pBlackboard.get()));
    pNodeArena[pNodePool.back()->getUniqueId()] = static_cast<uint32_t>((pNodePool.size() - 1));
    return pNodePool.back().get();
  }

  /// Create a new node edge returns its raw pointer
  Edge* buildEdge(Node* head, Node* tail)
  {
    // TODO add garbage collector: check if size is greater than max and, if so,
    // resize the pool
    pEdgePool.push_back(std::make_unique<Edge>(head, tail));
    pEdgeArena[pEdgePool.back()->getUniqueId()] = static_cast<uint32_t>((pEdgePool.size() - 1));
    return pEdgePool.back().get();
  }

  /// Returns the pointer to the node with given id
  Node* getNode(uint32_t nodeId) const
  {
    return pNodePool.at(pNodeArena.at(nodeId)).get();
  }

  /// Returns the pointer to the edge with given id
  Edge* getEdge(uint32_t edgeId) const
  {
    return pEdgePool.at(pEdgeArena.at(edgeId)).get();
  }

  /// Deletes the node with given id
  void deleteNode(uint32_t nodeId);

  /// Deletes the edge with given id
  void deleteEdge(uint32_t edgeId);

  /// Returns the pointer to the internal blackboard
  Blackboard::SPtr getBlackboard() const noexcept { return pBlackboard; }

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

  /// Blackboard instance shared with all nodes created
  /// in this arena
  Blackboard::SPtr pBlackboard{nullptr};

};

}  // namespace btsolver
