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
#include "bt_solver/all_different.hpp"
#include "bt_solver/bt_solver.hpp"
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

void runBehaviorTreeTest()
{
  using namespace btsolver;
  using namespace btsolver::cp;

  auto arena = std::make_unique<BehaviorTreeArena>();

  auto root = arena->buildNode<btsolver::Sequence>("sequence");
  auto log1 = arena->buildNode<btsolver::LogNode>("log_1");
  auto log2 = arena->buildNode<btsolver::LogNode>("log_2");
  auto log3 = arena->buildNode<btsolver::LogNode>("log_3");
  reinterpret_cast<btsolver::LogNode*>(log1)->setLog("log_1");
  reinterpret_cast<btsolver::LogNode*>(log2)->setLog("log_2");
  reinterpret_cast<btsolver::LogNode*>(log3)->setLog("log_3");
  reinterpret_cast<btsolver::Sequence*>(root)->addChild(log1->getUniqueId());
  reinterpret_cast<btsolver::Sequence*>(root)->addChild(log2->getUniqueId());
  reinterpret_cast<btsolver::Sequence*>(root)->addChild(log3->getUniqueId());

  btsolver::BehaviorTree bt(std::move(arena));
  bt.setEntryNode(root->getUniqueId());
  bt.run();

  Domain<BitmapDomain> domain(120, 130);
  std::cout << domain.minElement() << std::endl;
  std::cout << domain.maxElement() << std::endl;
  std::cout << domain.size() << std::endl;
  auto& it = domain.getIterator();
  while(!it.atEnd())
  {
    std::cout << it.value() << std::endl;
    it.moveToNext();
  }

  // Print last element "atEnd()"
  std::cout << it.value() << std::endl;

  // Reset the iterator
  it.reset();

  std::cout << "Remove 125 from domain\n";

  // Remove an element
  domain.removeElement(125);
  std::cout << domain.size() << std::endl;
  while(!it.atEnd())
  {
    std::cout << it.value() << std::endl;
    it.moveToNext();
  }
  std::cout << it.value() << std::endl;
  it.reset();

  std::cout << "Remove bounds 120, 121 and 130 from domain\n";

  // Remove lower and upper bounds
  domain.removeElement(120);
  domain.removeElement(121);
  domain.removeElement(130);
  std::cout << domain.minElement() << std::endl;
  std::cout << domain.maxElement() << std::endl;
  std::cout << domain.size() << std::endl;

  while(!it.atEnd())
  {
    std::cout << it.value() << std::endl;
    it.moveToNext();
  }
  std::cout << it.value() << std::endl;
  it.reset();
}


void runSolver()
{
  using namespace btsolver;
  using namespace btsolver::cp;

  BTSolver solver;

  // Create a simple model
  auto model = std::make_shared<Model>("basic_model");

  int maxVars{40};
  for (int idx{0}; idx < maxVars; ++idx)
  {
    model->addVariable(std::make_shared<Variable>("var_" + std::to_string(idx), 1, maxVars));
  }

  tools::Timer timer;

  // Build AllDifferent constraint
  auto allDiff = std::make_unique<AllDifferent>("AllDifferent");
  allDiff->setScope(model->getVariables());

  auto arena = std::make_unique<BehaviorTreeArena>();
  allDiff->builBehaviorTreePropagator(arena.get());

  std::cout << "Wallclock time (msec.): " << timer.getWallClockTimeMsec() << std::endl;

  return;

  // Set the model into the solver
  solver.setModel(model);

  // Build the relaxed BT
  auto bt = solver.buildRelaxedBT();

  // Build the exact BT starting from the relaxed BT
  solver.buildExactBT(bt);

  // Run the solver on the exact BT
  solver.setBehaviorTree(bt);

  // Run solver on the relaxed Behavior Tree
  solver.solve(1);
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
    runSolver();
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
