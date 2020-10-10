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
  using UPtr = std::unique_ptr<TDCompiler>;
  using SPtr = std::shared_ptr<TDCompiler>;

 public:
  TDCompiler(MDDProblem::SPtr problem, uint32_t width);

  /**
   * \brief compiles the MDD.
   * \return true if the MDD allows solutions, returns false otherwise.
   */
  bool compileMDD();

  /**
   * \brief sets the incumbent.
   */
  void setIncumbent(double incumbent) noexcept { pIncumbent = incumbent; }

  /**
   * \brief recompiles the MDD from the queue of stored states
   *        accumulated during previous compilations.
   * \arg[out] true if the MDD can be rebuild, false otherwise
   */
  bool rebuildMDDFromQueue();

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

  /// Returns the incumbent, i.e., the cost of the best solution found so far
  double getIncumbent() const noexcept { return pIncumbent; };

  /**
   * \brief given the current state, lower and upper bounds of the variable at that state,
   *        and the list of state on next layer, calculates the state that will be replaced
   *        and be part of the next layer's states.
   */
  DPState::ReplacementNodeList calculateNextLayerStates(uint32_t currLayer, uint32_t currNode,
                                                        int64_t lb, int64_t ub);
};

}  // namespace mdd
