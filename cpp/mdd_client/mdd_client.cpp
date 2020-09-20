//
// Copyright BTSolver 2020. All rights reserved.
//
// Entry point for the BTSolver client.
//
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
  int maxVars{10};
  for (int idx{0}; idx < maxVars; ++idx)
  {
    problem->addVariable(std::make_shared<Variable>(idx, idx, 1, maxVars));
  }

  // Create the MDD
  int32_t width{10};
  MDD mdd(problem, width);

  tools::Timer timer;

  // Build the relaxed MDD
  mdd.buildRelaxedMDD();

  std::cout << "Wallclock time (msec.): " << timer.getWallClockTimeMsec() << std::endl;
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
