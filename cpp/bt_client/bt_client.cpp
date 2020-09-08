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
    // Add entry point code here
    std::cout << "Framework in use: " << framework << std::endl;
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
