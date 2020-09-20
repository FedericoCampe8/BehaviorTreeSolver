//
// Copyright OptiLab 2020. All rights reserved.
//
// All different constraint based on BT optimization.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <vector>

#include <sparsepp/spp.h>

#include "mdd_optimization/edge.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS Node {
 public:
  using EdgeList = std::vector<Edge*>;
  using UPtr = std::unique_ptr<Node>;
  using SPtr = std::shared_ptr<Node>;

 public:
  /**
   * \brief Constructor, it does NOT take ownership of the given object
   */
  Node(uint32_t layer, Variable* variable = nullptr);

  /// Destructor: removes this node from the edges
  ~Node();

  const std::vector<int64_t>& getValues() const noexcept {
    return pVariable->getAvailableValues();
  }

  /// Returns this node's layer
  uint32_t getLayer() const noexcept {
    return pLayer;
  }

  /// Returns this node's unique identifier
  uint32_t getUniqueId() const noexcept { return pNodeId; }

  /// Adds an incoming edge
  void addInEdge(Edge* edge);

  /// Adds an outgoing edge
  void addOutEdge(Edge* edge);

  /// Removes the incoming edge at given position
  void removeInEdge(uint32_t position);

  /// Removes the outgoing edge at given position
  void removeOutEdge(uint32_t position);

  /// Removes the incoming edge at given edge instance
  void removeInEdgeGivenPtr(Edge* edge);

  /// Removes the outgoing edge at given edge instance
  void removeOutEdgeGivenPtr(Edge* edge);

  /// Returns the list of incoming edges
  const EdgeList& getInEdges() const noexcept {
    return pInEdges;
  }

  /// Returns the list of outgoing edges
  const EdgeList& getOutEdges() const noexcept {
    return pOutEdges;
  }

  /// Returns the (raw) pointer to the variable paired with this node
  Variable* getVariable() const noexcept {
    return pVariable;
  }

  void setOptimizationValue(double opt_value) noexcept {
    pOptimizationValue = opt_value;
  }

  double getOptimizationValue() const noexcept {
    return pOptimizationValue;
  }

  void setSelectedEdge(Edge *edge);

  Edge* getSelectedEdge() const noexcept {
    return pSelectedEdge;
  }

 private:
   static uint32_t kNextID;

 private:
  /// Unique identifier for this node
  uint32_t pNodeId{0};

  /// Layer this node is at
  uint32_t pLayer{0};

  /// Raw pointer to the variable that has the domain defining
  /// the labels of the outgoing edges on this node.
  /// @note the terminal node does not have any variable
  Variable* pVariable{nullptr};

  /// List of incoming edges
  EdgeList pInEdges;

  /// List of outgoing edges
  EdgeList pOutEdges;

  /// Sets storing incoming/outgoing edges for quick lookup
  spp::sparse_hash_set<uint32_t> pInEdgeSet;
  spp::sparse_hash_set<uint32_t> pOutEdgeSet;

  /// Optimization value on this node
  double pOptimizationValue{std::numeric_limits<double>::lowest()};

  Edge* pSelectedEdge{nullptr};
};

}  // namespace mdd
