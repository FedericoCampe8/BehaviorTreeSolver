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
#include "bt/branch.hpp"
#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "bt/std_node.hpp"
#include "cp/domain.hpp"
#include "cp/bitmap_domain.hpp"

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

void runBehaviorTree()
{
  using namespace btsolver;
  using namespace btsolver::cp;

  auto root = std::make_unique<btsolver::Sequence>("sequence");
  auto log1 = std::make_unique<btsolver::LogNode>("log_1");
  auto log2 = std::make_unique<btsolver::LogNode>("log_2");
  auto log3 = std::make_unique<btsolver::LogNode>("log_3");
  log1->setLog("log_1");
  log2->setLog("log_2");
  log3->setLog("log_3");
  root->addChild(std::move(log1));
  root->addChild(std::move(log2));
  root->addChild(std::move(log3));

  btsolver::BehaviorTree bt;
  bt.setEntryNode(std::move(root));
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
    runBehaviorTree();
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
