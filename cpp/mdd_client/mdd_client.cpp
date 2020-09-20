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

#include "mdd_optimization/variable.hpp"
#include "tools/timer.hpp"

namespace {

void runMDDOpt()
{
  using namespace mdd;

  tools::Timer timer;

  std::vector<int64_t> values = {1, 2, 3};
  auto var = std::make_shared<Variable>(0, 0, values);
  for (auto value : var->getAvailableValues())
  {
    std::cout << "Var val: " << value << std::endl;
  }

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
