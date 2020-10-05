//
// Copyright OptiLab 2020. All rights reserved.
//
// Top-Down compiler.
//

#pragma once

#include <memory>  // for std::unique_ptr

#include "mdd_optimization/arena.hpp"
#include "mdd_optimization/mdd_compiler.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/node.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief Top-Down compiler for compiling MDDs.
 *        The MDD to compile is represented by a single constraint which,
 *        in turn, is represented as a Dynamic Programming model.
 */
class SYS_EXPORT_CLASS TopDownCompiler : public MDDCompiler {
 public:
  using UPtr = std::unique_ptr<TopDownCompiler>;
  using SPtr = std::shared_ptr<TopDownCompiler>;

 public:
  TopDownCompiler(MDDProblem::SPtr problem);

  /**
   * \brief Compiles the MDD.
   * \param[in] mddGraph: the graph data structure reference to the MDD to build.
   * \param[in] arena: arena to build nodes and edges.
   * \param[int] the node pool used for branch & bound search.
   */
  void compileMDD(MDDCompiler::MDDGraph& mddGraph, Arena* arena,
                  MDDCompiler::NodePool& nodePool) override;

 private:
  /// Optimization problem to compile into an MDD
  MDDProblem::SPtr pProblem{nullptr};

  /// Pointer to the arena
  Arena* pArena{nullptr};

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

  /// Compiles the MDD starting from the root node
  void buildTopDownMDD(MDDGraph& mddGraph, NodePool& nodePool);

  /// Run merge nodes procedure for relaxed MDDs
  void mergeNodes(int layer, MDDGraph& mddGraph);

  /// Run nodes removal for restricted procedure
  void removeNodes(int layer, MDDGraph& mddGraph, NodePool& nodePool);
};

}  // namespace mdd
