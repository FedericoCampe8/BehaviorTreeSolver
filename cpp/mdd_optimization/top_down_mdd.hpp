//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD data structure optimized for top-down compilers.
//

#pragma once

#include <cstdint>  // for int64_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <string>
#include <utility>  // for std::pair
#include <vector>

#include <sparsepp/spp.h>

#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/mdd_constraint.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief class encapsulating an edge of the MDD.
 *        An edge can be active (i.e., connecting two edges)
 *        or not and it can have one or more domain values.
 *        If the edge has more than one value, it is a "parallel" edge.
 */
struct SYS_EXPORT_STRUCT MDDTDEdge {
  using UPtr = std::unique_ptr<MDDTDEdge>;

  MDDTDEdge();
  MDDTDEdge(int32_t tailLayer, int32_t tailIdx=-1, int32_t headIdx=-1);

  /**
   * \brief returns whether or not this is a parallel edge.
   */
  bool isParallel() const noexcept { return valuesList.size() > 1; }

  /**
   * \brief returns true if the first value on this edge is set.
   *        Returns false otherwise.
   */
  bool hasValueSet() const noexcept
  {
    return valuesList.at(0) < std::numeric_limits<int64_t>::max();
  }

  /**
   * \brief returns a string description of this edge.
   */
  std::string toString() const noexcept;

  /// Flag indicating whether or not this edge is active
  bool isActive{false};

  /// Layer of the tail node
  int32_t layer{-1};

  /// Tail node index
  int32_t tail{-1};

  /// Head node index
  int32_t head{-1};

  /// List of values on this edge
  std::vector<int64_t> valuesList;
};

/**
 * \brief Optimized version of MDD for top-down compilation.
 *        The MDD is represented as a matrix of states (layers x width)
 *        and a collection of edges per layer.
 *        Nodes are implicitly represented as indices of the states
 *        of a given layer.
 *        Edges are also pre-allocated for each layer.
 *        Each layer can have at most width x width edges.
 */
class SYS_EXPORT_CLASS TopDownMDD {
 public:
  /// List of all the state on a given layer
  using StateList = std::vector<DPState::UPtr>;

  using UPtr = std::unique_ptr<TopDownMDD>;
  using SPtr = std::shared_ptr<TopDownMDD>;

 public:
  /**
   * \brief builds a new Top-Down MDD
   *        with a number of layers equal to the number of variables plus one
   *        (i.e., the root) and the specified width.
   * \note the problem must contain ONLY one MDD constraint.
   */
  TopDownMDD(MDDProblem::SPtr problem, uint32_t width);

  /**
   * \brief returns the number of layers in the MDD.
   */
  uint32_t getNumLayers() const noexcept { return pNumLayers; };

  /**
   * \brief returns the maximum width of this MDD.
   */
  uint32_t getMaxWidth() const noexcept { return pMaxWidth; };

  /**
   * \brief rebuilds the MDD from the given state and up to the given state.
   */
  void rebuildMDDFromState(DPState::UPtr state);

  /**
   * \brief resets all the MDD as if it was just built.
   */
  void resetGraph();

  /**
   * \brief returns the state at given layer for the given node index.
   */
  DPState* getNodeState(uint32_t layerIdx, uint32_t nodeIdx) const
  {
    return pMDDStateMatrix.at(layerIdx).at(nodeIdx).get();
  }

  /**
   * \brief returns the edge at given layer with given tail and head
   */
  MDDTDEdge* getEdgeMutable(uint32_t layerIdx,  uint32_t tailIdx, uint32_t headIdx) const;

  /**
   * \brief returns the variables paired to a given layer.
   */
  Variable* getVariablePerLayer(uint32_t layerIdx) const
  {
    return getVariablesList().at(layerIdx).get();
  }

  /**
   * \brief returns the list of states on the specified layer.
   */
  StateList* getStateListMutable(uint32_t layerIdx)
  {
    return &pMDDStateMatrix[layerIdx];
  }

  /**
   * \brief returns the index of the first default state on the given level.
   *        If no default states are present, the index is equal to width
   */
  uint32_t getIndexOfFirstDefaultStateOnLayer(uint32_t layerIdx) const;

  /**
   * \brief returns the list of active edges that are on layer "layerIdx" and
   *        having "headIdx" as head node for "layerIdx".
   */
  std::vector<MDDTDEdge*> getEdgeOnHeadMutable(uint32_t layerIdx, uint32_t headIdx) const;

  /**
   * \brief returns the list of active edges on the given layer.
   */
  std::vector<MDDTDEdge*> getActiveEdgesOnLayer(uint32_t layerIdx) const;

  /**
   * \brief replace the state "nodeIdx" at layer "layerIdx" with "state".
   * \return the replaced state.
   */
  DPState::UPtr replaceState(uint32_t layerIdx, uint32_t nodeIdx, DPState::UPtr state);

  /**
   * \brief disable the edge at given layer with specified tail and head.
   */
  void disableEdge(uint32_t layerIdx, uint32_t tailIdx, uint32_t headIdx);

  /**
   * \brief enable the edge at given layer with specified tail and head.
   */
  void enableEdge(uint32_t layerIdx, uint32_t tailIdx, uint32_t headIdx);

  /**
   * \brief sets the value on the specified edge.
   * \note sets the FIRST value on the edge, i.e., as if the edge was NOT a parallel edge.
   */
  void setEdgeValue(uint32_t layerIdx,  uint32_t tailIdx, uint32_t headIdx, int64_t val);

  /**
   * \brief returns true if the node (head) at given layer is reachable.
   *        returns false otherwise.
   */
  bool isReachable(uint32_t layerIdx, uint32_t headIdx) const;

  /**
   * \brief removes the state (node head) at given layer.
   */
  void removeState(uint32_t layerIdx, uint32_t headIdx);

  /**
   * \brief prints a JPEG representation of this MDD to the given file "name.dot".
   *        The file can be read by graphviz.
   *        For more information, visit https://graphviz.org/about/.
   * \note prints only active edges.
   */
  void printMDD(const std::string& outFileName) const;

 private:
  /// Ordered list of edges
  using EdgeList = std::vector<MDDTDEdge::UPtr>;

  /// Matrix of edges: one edge list per layer
  using LayerEdgeList = std::vector<EdgeList>;

  /// Matrix of states in the MDD
  using MDDStateMatrix = std::vector<StateList>;

 private:
  /// Pointer to the optimization problem
  MDDProblem::SPtr pProblem{nullptr};

  /// MDD max width
  uint32_t pMaxWidth{1};

  /// MDD number of layers
  uint32_t pNumLayers{0};

  /// Collection of edges in the MDD.
  /// Edges are stored "per-layer".
  /// For example, layer zero contains all edges that have
  /// the root as tail node
  LayerEdgeList pLayerEdgeList;

  /// Collection of states in the MDD
  MDDStateMatrix pMDDStateMatrix;

  /// Map of tracking the index of the first default state on each layer
  spp::sparse_hash_map<uint32_t, uint32_t> pStartDefaultStateIdxOnLevel;

  /// Pre-allocate all edges in the MDD.
  /// @note edges are allocated on the heap
  void allocateEdges();

  /// Pre-allocate all states in the MDD.
  /// @note states are allocated on the heap
  void allocateStates(MDDConstraint::SPtr con);

  /// Returns the list of constraints in the problem
  const MDDProblem::ConstraintsList& getConstraintsList() const noexcept
  {
    return pProblem->getConstraints();
  }

  /// Returns the list of variables in the problem
  const MDDProblem::VariablesList& getVariablesList() const noexcept
  {
    return pProblem->getVariables();
  }

  /// Returns true if the given state is a leaf state.
  /// Returns false otherwise.
  /// A leaf node is a node that is reachable and doesn't have active outgoing edges
  bool isLeafState(uint32_t layerIdx, uint32_t nodeIdx) const;
};

}  // namespace mdd
