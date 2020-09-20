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

#include "mdd_optimization/edge.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS Node {
 public:
  using UPtr = std::unique_ptr<Node>;
  using SPtr = std::shared_ptr<Node>;

 public:
  /**
   * \brief Constructor, it does NOT take ownership of the given object
   */
  Node(Variable *variable, uint32_t layer);

  const std::vector<int64_t>& getValues() const noexcept {
    return pVariable->getAvailableValues();
  }

  /// Returns this node's layer
  uint32_t getLayer() const noexcept {
    return pLayer;
  }

  /// Adds an incoming edge
  void addInEdge(Edge *edge);

  /// Adds an outgoing edge
  void addOutEdge(Edge *edge);

  /// Removes the incoming edge at given position
  void removeInEdge(uint32_t position);

  /// Removes the outgoing edge at given position
  void removeOutEdge(uint32_t position);

  /// Returns the list of incoming edges
  const std::vector<Edge*>& get_in_edges() const noexcept {
    return pInEdges;
  }

  /// Returns the list of outgoing edges
  const std::vector<Edge*>& getOutEdges() const noexcept {
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
  uint32_t pLayer{0};

  Variable *pVariable {nullptr};

  /// List of incoming edges
  std::vector<Edge*> pInEdges;

  /// List of outgoing edges
  std::vector<Edge*> pOutEdges;

  /// Optimization value on this node
  double pOptimizationValue{std::numeric_limits<double>::lowest()};

  Edge* pSelectedEdge{nullptr};
};

}  // namespace mdd
