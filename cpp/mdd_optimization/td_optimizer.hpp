//
// Copyright OptiLab 2020. All rights reserved.
//
// Branch & bound optimization solver based on top-down MDD compilation.
//

#pragma once

#include <cstdint>  // for int32_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr
#include <vector>

#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/td_compiler.hpp"
#include "mdd_optimization/top_down_mdd.hpp"
#include "system/system_export_defs.hpp"
#include "tools/timer.hpp"

namespace mdd {

class SYS_EXPORT_CLASS TDMDDOptimizer {
 public:
  using UPtr = std::unique_ptr<TDMDDOptimizer>;
  using SPtr = std::shared_ptr<TDMDDOptimizer>;

 public:
  /**
   * \brief builds a new top-down MDD optimizer for the given problem.
   */
  TDMDDOptimizer(MDDProblem::SPtr problem);

  /**
   * \ brief runs the optimization process compiling
   *         MDDs with specified width and applying
   *         branch & bound on the compiled MDDs.
   */
  void runOptimization(uint32_t width, uint64_t timeoutMsec=std::numeric_limits<uint64_t>::max());

  /**
   * \brief sets the maximum number of solutions.
   * \note by default it finds the first solution.
   */
  void setMaxNumSolutions(uint64_t maxNumSolutions) noexcept { pNumMaxSolutions = maxNumSolutions; }

  /**
   * \brief returns the (best) cost of the solution found so far.
   */
  double getBestCost() const noexcept { return pBestCost; }

  /**
   * \breif returns the number of solutions found so far.
   */
  uint64_t getNumSolutions() const noexcept { return pNumSolutionsCtr; }

  /**
   * \brief Prints a JPEG representation of this MDD to the given file name.dot.
   *        The file can be read by graphviz.
   *        For more information, visit https://graphviz.org/about/
   */
  void printMDD(const std::string& outFileName);

 private:
  /// Max width of the MDD
  uint32_t pMaxWidth{0};

  /// Optimization model/problem to solve
  MDDProblem::SPtr pProblem{nullptr};

  /// List of all layers with nodes in this MDD
  TDCompiler::UPtr pCompiler;

  /// Counter on the number of solutions
  uint64_t pNumSolutionsCtr{0};

  /// Maximum number of solution
  uint64_t pNumMaxSolutions{std::numeric_limits<uint64_t>::max()};

  /// Timer that starts on construction
  tools::Timer::SPtr pTimer;

  /// Cost value of the best solution found so far
  double pBestCost{std::numeric_limits<double>::max()};

  /// Updates the solution cost
  void updateSolutionCost(double cost);

  /// Runs DFS to find the minimum value on the MDD starting from
  /// the given node
  void dfsRec(TopDownMDD* mddGraph, DPState* state, std::vector<int64_t>& path, double& bestCost,
              const uint32_t currLayer, double cost=0.0);
};

}  // namespace mdd
