//
// Copyright OptiLab 2020. All rights reserved.
//
// Branch & bound optimization solver based on MDD compilation.
//

#pragma once

#include <cstdint>  // for int32_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <vector>

#include "mdd_optimization/arena.hpp"
#include "mdd_optimization/mdd_compiler.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/node.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS MDDOptimizer {
 public:
  using UPtr = std::unique_ptr<MDDOptimizer>;
  using SPtr = std::shared_ptr<MDDOptimizer>;

 public:
  /**
   * \brief builds a new MDD optimizer for the given problem.
   */
  MDDOptimizer(MDDProblem::SPtr problem);

  /**
   * \ brief runs the optimization process compiling
   *         MDDs with specified width and applying
   *         branch & bound on the compiled MDDs.
   */
  void runOptimization(int32_t width);

  /// Prints a JPEG representation of this MDD to the given file name.dot.
  /// The file can be read by graphviz.
  /// For more information, visit https://graphviz.org/about/
  void printMDD(const std::string& outFileName);

 private:
  /// Max width of the MDD
  int32_t pMaxWidth{-1};

  /// Pointer to the root node of the optimal MDD
  Node* pRootNode{nullptr};

  /// Optimization model/problem to solve
  MDDProblem::SPtr pProblem{nullptr};

  /// List of all layers with nodes in this MDD
  MDDCompiler::MDDGraph pMDDGraph;

  /// Pointer to the arena for building MDD objects
  Arena::UPtr pArena{nullptr};

  /// Pointer to the compiler to use for optimization
  MDDCompiler::UPtr pMDDCompiler{nullptr};

  /// Cost value of the best solution found so far
  double pBestCost{std::numeric_limits<double>::max()};

  /// Builds, sets and returns the root of the MDD
  void buildRootMDD();

  /// Rebuild the MDD from the root up to the given node
  void rebuildMDDUpToNode(Node* node);

  /// Runs DFS to find the minimum value on the MDD starting from
  /// the given node
  void dfsRec(Node* currNode, double& bestCost, const uint32_t maxLayer,
              double cost=0.0, Node* prevNode=nullptr, bool debug=false);
};

}  // namespace mdd
