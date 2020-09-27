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
#include <unordered_map>

#include <sparsepp/spp.h>

#include "mdd_optimization/dp_model.hpp"
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

  /// Initialized the domain of this node based on its variable
  void initializeNodeDomain();

  /// Returns this node's values
  const std::vector<int64_t>& getValues() const noexcept { return pNodeDomain; }

  /// Returns the pointer to this node's values
  std::vector<int64_t>* getValuesMutable() noexcept { return &pNodeDomain; }

  /// Returns this node's layer
  uint32_t getLayer() const noexcept { return pLayer; }

  /// Returns this node's unique identifier
  uint32_t getUniqueId() const noexcept { return pNodeId; }

  /// Returns true if this node is a leaf node.
  /// Returns false otherwise
  bool isLeaf() const noexcept { return getVariable() == nullptr; }

  /// Returns true if this is the root node, returns false otherwise
  bool isRootNode() const noexcept { return pInEdges.empty(); }

  /// Returns true if this is the terminal node, returns false otherwise
  bool isTerminalNode() const noexcept { return isLeaf(); }

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

  /// Returns this node DP state
  DPState* getDPState() const noexcept { return pDPState.get(); }

  /// Resets the internal DP state to the default one
  void setDefaultDPState()
  {
    pIsDPStateChanged = false;
    pDPState = pDefaultDPState;
  }

  /// Resets the internal DP state
  void resetDPState(DPState::SPtr dpState) noexcept
  {
    pIsDPStateChanged = true;
    pDPState = dpState;
  }

  /// Returns true if the nodes contains the original/default DP state.
  /// Returns false otherwise (e.g., the DP state has been reset)
  bool hasDefaultDPState() const noexcept { return !pIsDPStateChanged; }

  /// Returns the list of incoming edges
  const EdgeList& getInEdges() const noexcept { return pInEdges; }

  /// Returns the list of outgoing edges
  const EdgeList& getOutEdges() const noexcept {
    return pOutEdges;
  }

  /// Return the path to node
  const std::unordered_map< Edge*, std::vector< EdgeList > > getIncomingPaths() const noexcept { return pIncomingPathsForEdge; }

  /// Returns the (raw) pointer to the variable paired with this node
  Variable* getVariable() const noexcept { return pVariable; }

  void setOptimizationValue(double optValue) noexcept {
    pOptimizationValue = optValue;
  }

  double getOptimizationValue() const noexcept {
    return pOptimizationValue;
  }

  /// Sets the selected solution edge for this node
  void setSelectedEdge(Edge *edge);

  /// Returns the selected solution edge for this node
  Edge* getSelectedEdge() const noexcept { return pSelectedEdge; }

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

  /// Path from root to node
  std::unordered_map< Edge*, std::vector< EdgeList > > pIncomingPathsForEdge;

  /// Sets storing incoming/outgoing edges for quick lookup
  spp::sparse_hash_set<uint32_t> pInEdgeSet;
  spp::sparse_hash_set<uint32_t> pOutEdgeSet;

  /// Domain/state paired with this node.
  /// This is used on filtering compilation
  std::vector<int64_t> pNodeDomain;

  /// Flag indicating whether or not the original DP state has been changed
  bool pIsDPStateChanged{false};

  /// The DP state associated with this BT state
  DPState::SPtr pDPState{nullptr};

  /// The default DP state
  DPState::SPtr pDefaultDPState{nullptr};

  /// Optimization value on this node
  double pOptimizationValue{std::numeric_limits<double>::lowest()};

  /// Selected edge on solution
  Edge* pSelectedEdge{nullptr};
};

}  // namespace mdd
