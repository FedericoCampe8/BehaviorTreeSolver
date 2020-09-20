//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for MDD.
//

#pragma once

#include <cstdint>  // for int32_t
#include <memory>   // for std::unique_ptr
#include <vector>

#include "mdd_optimization/arena.hpp"
#include "mdd_optimization/edge.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/node.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS MDD {
 public:
  /**
   * \brief Type of algorithm to be used to build the MDD
   */
  enum class MDDConstructionAlgorithm {
    /// Build the MDD by separation
    Separation = 0,
    /// Build the MDD using the Top-Down approach
    TopDown
  };

  /// List of all the (pointers to the) nodes in an MDD layer
  using NodesLayerList = std::vector<Node*>;

  /// List of all the layers of the MDD
  using MDDLayersList = std::vector<NodesLayerList>;

  using UPtr = std::unique_ptr<MDD>;

 public:
  MDD(MDDProblem::SPtr problem, int32_t width);

  /// Builds the relaxed MDD w.r.t. the given problem.
  /// Given n variables it builds the MDD as
  ///
  ///  [ r ]           L_1
  ///    |     Dx_1
  ///  [u_1]           L_2
  ///    |
  ///   ...
  ///    |     Dx_j
  ///  [u_j]           L_j+1
  ///    |     Dx_j+1
  ///   ...
  ///    |     Dx_n
  ///  [ t ]           L_n+1
  ///
  /// Where r is the root node, t is the terminal node
  /// and each edge is a parallel edge with values equal to the
  /// domain of the correspondent variable.
  /// @note in the MDD papers they start counting layers from 1.
  /// Returns the pointer to the root node
  Node* buildRelaxedMDD();

  /// Enforces the constraints of the problem onto the given (relaxed) MDD
  void enforceConstraints(Node* relaxedMDD, MDDConstructionAlgorithm algorithmType);

  const MDDLayersList& getNodesPerLayer() const noexcept
  {
    return pNodesPerLayer;
  }

private:
  /// Max width of the MDD
  int32_t pMaxWidth{-1};

  /// Optimization model/problem to solve
  MDDProblem::SPtr pProblem{nullptr};

  /// List of all layers with nodes in this MDD
  MDDLayersList pNodesPerLayer;

  /// Pointer to the arena for building MDD objects
  Arena::UPtr pArena{nullptr};

  /// Pointer to the root node for the MDD
  Node* pRootNode{nullptr};

  /// Pointer to the terminal node for the MDD
  Node* pTerminalNode{nullptr};

  /// Expands vertically the given node to build a relaxed MDD
  Node* expandNode(Node* node);

  /// Runs the separation algorithm on the given MDD w.r.t. the constraint in the problem
  void runSeparationProcedure(Node* root);

  /// Runs the top-down algorithm on the given MDD w.r.t. the constraint in the problem
  void runTopDownProcedure(Node* root);
};

}  // namespace mdd
