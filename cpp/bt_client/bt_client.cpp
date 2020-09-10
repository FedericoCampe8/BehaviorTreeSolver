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

#include "bt/branch.hpp"
#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "bt/std_node.hpp"

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
  btsolver::Sequence sequence("sequence");
  auto log1 = std::make_unique<btsolver::LogNode>("log_1");
  auto log2 = std::make_unique<btsolver::LogNode>("log_2");
  auto log3 = std::make_unique<btsolver::LogNode>("log_3");
  log1->setLog("log_1");
  log2->setLog("log_2");
  log3->setLog("log_3");
  sequence.addChild(std::move(log1));
  sequence.addChild(std::move(log2));
  sequence.addChild(std::move(log3));

  auto status = btsolver::NodeStatus::kActive;
  while (status == btsolver::NodeStatus::kActive)
  {
    sequence.tick();
    status = sequence.getResult();
  }
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
