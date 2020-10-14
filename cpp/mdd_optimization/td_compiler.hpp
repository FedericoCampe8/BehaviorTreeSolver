//
// Copyright OptiLab 2020. All rights reserved.
//
// Top-Down compiler.
//

#pragma once

#include <limits>  // for std::numeric_limits
#include <memory>  // for std::unique_ptr

#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/top_down_mdd.hpp"
#include "mdd_optimization/variable.hpp"

#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief Top-Down compiler for compiling MDDs.
 *        The MDD to compile is represented by a single constraint which,
 *        in turn, is represented as a Dynamic Programming model.
 */
class SYS_EXPORT_CLASS TDCompiler {
 public:
  /**
   * \brief compilation mode for the top-down compiler.
   */
  enum class CompilationMode {
    Relaxed = 1,
    Restricted
  };

  using UPtr = std::unique_ptr<TDCompiler>;
  using SPtr = std::shared_ptr<TDCompiler>;

 public:
  TDCompiler(MDDProblem::SPtr problem, uint32_t width);

  /**
   * \brief compiles the MDD.
   * \return true if the MDD allows solutions, returns false otherwise.
   */
  bool compileMDD(CompilationMode compilationMode);

  /**
   * \brief sets the incumbent.
   */
  void setIncumbent(double incumbent) noexcept { pIncumbent = incumbent; }

  /**
   * \brief returns the compiled MDD.
   */
  TopDownMDD* getMDDMutable() const noexcept { return pMDDGraph.get(); }

  /**
   * \brief prints the MDD to a viz file with given name.
   */
  void printMDD(const std::string& outFileName) const { pMDDGraph->printMDD(outFileName); }

 private:
  /// Optimization problem to compile into an MDD
  MDDProblem::SPtr pProblem{nullptr};

  /// MDD data structure
  TopDownMDD::UPtr pMDDGraph{nullptr};

  double pIncumbent{std::numeric_limits<double>::max()};

  /**
   * \brief builds the MDD using top-down compilation.
   */
  bool buildMDD(CompilationMode compilationMode);

  /**
   * \brief returns the incumbent, i.e., the cost of the best solution found so far.
   */
  double getIncumbent() const noexcept { return pIncumbent; };

  /**
   * \brief given the current node, replace all states on next layer with
   *        the states that are obtained by "currNode" using its function "next".
   *        States are replaced only if their cost is lower than current cost (heuristic).
   * \return the list of edges that should be activated due to state replacement.
   *         For each edge, there is a Boolean flag: if true, every current active edge leading
   *         to that node should be deactivated (the state has been completely replaced).
   *         If false, any active edge should remain active and a new active edge should be added
   *         (the state is shared).
   */
  std::vector<std::pair<MDDTDEdge*, bool>> restrictNextLayerStatesFromNode(uint32_t currLayer,
                                                                           uint32_t currNode,
                                                                           int64_t lb, int64_t ub);

  /**
   * \brief given the current node, merge all states on next layer with
   *        the states that are obtained by "currNode" using its function "next".
   *        States are replaced only if their cost is lower than current cost (heuristic).
   * \return the list of edges that should be activated due to state replacement.
   *         For each edge, there is a Boolean flag: if true, every current active edge leading
   *         to that node should be deactivated (the state has been completely replaced).
   *         If false, any active edge should remain active and a new active edge should be added
   *         (the state is shared).
   */
  std::vector<std::pair<MDDTDEdge*, bool>> relaxNextLayerStatesFromNode(uint32_t currLayer,
                                                                        uint32_t currNode,
                                                                        int64_t lb, int64_t ub);
};

}  // namespace mdd
