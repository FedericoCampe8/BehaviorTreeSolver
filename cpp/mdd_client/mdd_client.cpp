//
// Copyright BTSolver 2020. All rights reserved.
//
// Entry point for the BTSolver client.
//
#include <exception>
#include <iostream>
#include <limits>  // for std::numeric_limits
#include <memory>
#include <string>
#include <vector>

#include "mdd_optimization/all_different.hpp"
#include "mdd_optimization/mdd.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/variable.hpp"
#include "tools/timer.hpp"

namespace {

void runMDDOpt()
{
  using namespace mdd;

  // Create the MDD problem
  auto problem = std::make_shared<MDDProblem>();

  // Add the list of variables
  int32_t maxVars{10};
  for (int idx{0}; idx < maxVars; ++idx)
  {
    problem->addVariable(std::make_shared<Variable>(idx, idx, 1, maxVars));
  }

  // Create the constraint
  auto allDiff = std::make_shared<AllDifferent>();
  allDiff->setScope(problem->getVariables());
  problem->addConstraint(allDiff);

  // Create the MDD
  int32_t width{std::numeric_limits<int32_t>::max()};
  width = 3;
  MDD mdd(problem, width);

  tools::Timer timer;

  // Enforce all the constraints on the MDD
  //MDD::MDDConstructionAlgorithm::Separation
  mdd.enforceConstraints(MDD::MDDConstructionAlgorithm::SeparationWithIncrementalRefinement);
  std::cout << "Wallclock time enforce constraints (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;

  timer.reset();
  timer.start();
  auto solution = mdd.maximize();
  for (int idx = 0; idx < solution.size(); idx++)
  {
      auto edge = solution[idx];
      std::cout << edge->getTail()->getLayer() << " - " << edge->getHead()->getLayer() << ": " <<
              edge->getValue() << std::endl;
  }
  std::cout << "Wallclock time solution (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {

  // Run the MDD
  try
  {
    // TODO Add entry point code here
    runMDDOpt();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << "Undefined error" << std::endl;
    return 2;
  }

  return 0;
}
