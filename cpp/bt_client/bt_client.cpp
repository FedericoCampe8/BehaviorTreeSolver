//
// Copyright BTSolver 2020. All rights reserved.
//
// Entry point for the BTSolver client.
//

#include <getopt.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "bt/behavior_tree.hpp"
#include "bt/behavior_tree_arena.hpp"
#include "bt/branch.hpp"
#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "bt/std_node.hpp"
#include "bt_optimization/all_different.hpp"
#include "bt_optimization/bt_solver.hpp"
#include "cp/bitmap_domain.hpp"
#include "cp/domain.hpp"
#include "cp/model.hpp"
#include "cp/variable.hpp"
#include "tools/timer.hpp"

extern int optind;

namespace {

void printHelp(const std::string& programName) {
  std::cerr << "Usage: " << programName << " [options]"
      << std::endl
      << "options:" << std::endl
      << "  --framework|-f     Specifies the framework to be used: "
          "CP, ORTools (default CP).\n"
      << "  --help|-h          Print this help message."
      << std::endl;
}  // printHelp

void runSolverOpt()
{
  using namespace btsolver;
  using namespace btsolver::optimization;

  BTOptSolver solver;

  // Create a simple model
  auto model = std::make_shared<cp::Model>("basic_model");

  // Switch num. variables for testing
  int maxVars{10};
  for (int idx{0}; idx < maxVars; ++idx)
  {
    model->addVariable(std::make_shared<cp::Variable>("var_" + std::to_string(idx), 1, maxVars));
  }

  tools::Timer timer;

  // Set the model into the solver
  solver.setModel(model);

  // Build the relaxed BT
  auto bt = solver.buildRelaxedBT();
  std::cout << "Wallclock time building relaxed BT (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;

  auto allDiff = std::make_shared<AllDifferent>(bt->getArenaMutable(), "AllDifferent");
  allDiff->setScope(model->getVariables());
  model->addConstraint(allDiff);

  timer.reset();
  timer.start();
  solver.separateBehaviorTree(bt);

  // Run the solver on the behavior tree
  solver.setBehaviorTree(bt);

  // Run solver on the relaxed Behavior Tree
  solver.solve(1);

  std::cout << "Wallclock time (msec.): " << timer.getWallClockTimeMsec() << std::endl;
}


}  // namespace

int main(int argc, char* argv[]) {

  char optString[] = "hf:";
  struct option longOptions[] =
  {
      { "framework", required_argument, NULL, 'f' },
      { "help", no_argument, NULL, 'h' },
      { 0, 0, 0, 0 }
  };

  // Parse options
  int opt;
  std::string framework = "CP";
  while (-1 != (opt = getopt_long(argc, argv, optString, longOptions, NULL)))
  {
    switch (opt)
    {
      case 'f':
      {
        framework = std::string(optarg);
        break;
      }
      case 'h':
      default:
        printHelp(argv[0]);
        return 0;
    }
  }  // while

  // Print some info
  try
  {
    // TODO Add entry point code here
    runSolverOpt();
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