//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for MDD compilers.
//

#pragma once

#include <limits>  // for std::numeric_limits
#include <memory>  // for std::unique_ptr
#include <queue>   // for std::priority_queue
#include <vector>

#include "mdd_optimization/arena.hpp"
#include "mdd_optimization/node.hpp"
#include "mdd_optimization/mdd_problem.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

class SYS_EXPORT_CLASS CompareNodes {
public:
  CompareNodes()
  {
  }

  bool operator() (const mdd::Node* lhs, const mdd::Node* rhs) const
  {
    return (lhs->getDPState()->cumulativeCost() > lhs->getDPState()->cumulativeCost());
  }
};

class SYS_EXPORT_CLASS MDDCompiler {
 public:
  /**
   * \brief Type of algorithm to be used to build the MDD
   */
  enum class MDDConstructionAlgorithm {
    /// Build the MDD by separation
    Separation = 0,
    /// Build a relaxed MDD using the separation and refinement algorithms
    SeparationWithIncrementalRefinement,
    /// Build the MDD using the Top-Down approach
    TopDown,
    /// Build a restricted MDD using the Top-Down approach
    RestrictedTopDown,
    /// Build the MDD by filtering and splitting a relaxed MDD
    Filtering
  };

  enum class MDDCompilationType {
    /// Build a complete MDD
    Exact = 0,
    /// Build a relaxed MDD
    Relaxed,
    /// Build a restricted MDD
    Restricted
  };

  /**
   * \brief strategy to select nodes to remove during
   *        restricted compilation
   */
  enum class RestrictedNodeSelectionStrategy {
    /// Remove nodes right to left on each layer
    RightToLeft = 0,
    /// Remove nodes left to right on each layer
    LeftToRight,
    /// Remove nodes randomly
    Random,
    /// Removes nodes following the cumulative cost on each state
    /// from the most expensive to the cheapest
    CumulativeCost
  };

  /// List of all the (pointers to the) nodes in an MDD layer
  using NodesLayerList = std::vector<Node*>;

  /// Ordered List of all the layers of the MDD
  using MDDGraph = std::vector<NodesLayerList>;

  /// Pool of nodes for branc and bound search
  using NodePool = std::vector<std::priority_queue<Node*, std::vector<Node*>, CompareNodes>>;

  using UPtr = std::unique_ptr<MDDCompiler>;
  using SPtr = std::shared_ptr<MDDCompiler>;

 public:
  virtual ~MDDCompiler() {}

  /// Returns the type of this compiler
  MDDConstructionAlgorithm getCompilerType() const noexcept { return pCompilerType; }

  /// Sets the compilation type.
  /// @note default compilation type is restricted
  void setCompilationType(MDDCompilationType compilationType) noexcept
  {
    pCompilationType = compilationType;
  }

  /// Returns the compilation type
  MDDCompilationType getCompilationType() const noexcept { return pCompilationType; }

  /**
   * \brief Sets the node removal strategy to be used during restricted compilation.
   *        Note: CumulativeCost is set by default
   * \param[in] strat the strategy to set for nodes removal
   */
  void setNodesRemovalStrategy(RestrictedNodeSelectionStrategy strat) noexcept
  {
    pNodesRemovalStrategy = strat;
  }

  /// Returns the nodes removal strategy
  RestrictedNodeSelectionStrategy getNodesRemovalStrategy() const noexcept
  {
    return pNodesRemovalStrategy;
  }

  /// Sets the flag to force state equivalence check and merge during compilation
  void forceStateEquivalenceCheckAndMerge(bool eqCheck=true) noexcept
  {
    pForceStateEquivalenceMerge = eqCheck;
  }

  /// Sets the incumbent
  void setIncumbent(double incumbent) noexcept { pBestCost = incumbent; }

  /**
   * \brief Sets the maximum width for the MDD.
   *        Note that this is needed only for relaxed or restricted compilations.
   * \param[in] maxWidth the maximum width of the MDD to compile
   */
  void setMaxWidth(int32_t maxWidth) noexcept { pMaxWidth = maxWidth; }

  /**
   * \brief Compiles the MDD.
   * \param[in] mddGraph: the graph data structure reference to the MDD to build.
   * \param[in] arena: arena to build nodes and edges.
   * \param[in] the node pool used for branch & bound search.
   */
  virtual void compileMDD(MDDGraph& mddGraph, Arena* arena, NodePool& nodePool) = 0;

 protected:
  MDDCompiler(MDDConstructionAlgorithm compilerType)
 : pCompilerType(compilerType)
 {
 }

  /// Returns the maximum width of the MDD to build
  int32_t getMaxWidth() const noexcept { return pMaxWidth; }

  /// Returns the (best) incumbent
  double getBestCost() const noexcept { return pBestCost; }

  /// Returns the flag indicating whether the state equivalence check
  /// and merge is enabled or not
  bool isStateEquivalenceCheckAndMergeEnabled() const noexcept
  {
    return pForceStateEquivalenceMerge;
  }

 private:
  /// Type of this compiler
  MDDConstructionAlgorithm pCompilerType{MDDConstructionAlgorithm::TopDown};

  /// Compilation type
  MDDCompilationType pCompilationType{MDDCompilationType::Restricted};

  /// Nodes removal strategy
  RestrictedNodeSelectionStrategy pNodesRemovalStrategy{
    RestrictedNodeSelectionStrategy::CumulativeCost};

  /// Flag indicating whether or not state equivalence check and merge
  /// should be applied
  bool pForceStateEquivalenceMerge{false};

  /// Maximum width of the MDD to build
  int32_t pMaxWidth{std::numeric_limits<int32_t>::max()};

  /// Best incumbent found so far
  double pBestCost{std::numeric_limits<double>::max()};
};

}  // namespace mdd
