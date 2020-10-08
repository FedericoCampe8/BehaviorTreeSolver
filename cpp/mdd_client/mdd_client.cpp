//
// Copyright BTSolver 2020. All rights reserved.
//
// Entry point for the BTSolver client.
//
#include <cstdint>    // for int64_t
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>     // for std::numeric_limits
#include <memory>
#include <string>
#include <vector>

#include <rapidjson/document.h>

#include "mdd_optimization/all_different.hpp"
#include "mdd_optimization/among.hpp"
#include "mdd_optimization/mdd.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/tsppd.hpp"
#include "mdd_optimization/variable.hpp"
#include "tools/timer.hpp"

namespace {

// void runTSPPD()
// {
//   using namespace mdd;

//   std::string instancePath{"../cpp/mdd_client/data/grubhub-02-0.json"};
//   std::ifstream datafile(instancePath);
//   std::string dataString((std::istreambuf_iterator<char>(datafile)),
//                          (std::istreambuf_iterator<char>()));

//   // Parse the inputs into Json
//   rapidjson::Document dataDoc;
//   dataDoc.Parse(dataString.c_str());

//   // Parse the nodes
//   const auto& nodesJson = dataDoc["nodes"];
//   int numVars = static_cast<int>(nodesJson.Size());

//   // Parse the distance matrix
//   const auto& edgesJson = dataDoc["edges"];
//   std::vector<std::vector<int64_t>> costMatrix;
//   costMatrix.resize(edgesJson.Size());
//   for (rapidjson::SizeType idx{0}; idx < edgesJson.Size(); ++idx)
//   {
//     const auto& resObj = edgesJson[idx].GetArray();
//     auto& row = costMatrix[idx];
//     row.resize(resObj.Size());
//     for (rapidjson::SizeType idx2{0}; idx2 < resObj.Size(); ++idx2)
//     {
//       row[idx2] = resObj[idx2].GetInt64();
//     }
//   }

//   // Create the MDD problem
//   auto problem = std::make_shared<MDDProblem>();
//   problem->setMinimization();

//   // First variable is always the node "+0"
//   problem->addVariable(std::make_shared<Variable>(0, 0, 0, 0));

//   std::vector<int64_t> pickupNode;
//   std::vector<int64_t> deliveryNode;
//   for (int idx{1}; idx < numVars-1; ++idx)
//   {
//     if (((idx+1) % 2) == 0)
//     {
//       pickupNode.push_back(idx + 1);
//     }

//     if (((idx+2) % 2) == 1)
//     {
//       deliveryNode.push_back(idx + 2);
//     }
//     problem->addVariable(std::make_shared<Variable>(idx, idx, 2, numVars-1));
//   }

//   // Last variable is always the node "-0"
//   problem->addVariable(std::make_shared<Variable>(numVars-1, numVars-1, 1, 1));

//   // Create the constraint
//   auto tsppd = std::make_shared<TSPPD>(pickupNode, deliveryNode, costMatrix);
//   tsppd->setScope(problem->getVariables());
//   problem->addConstraint(tsppd);

//   tools::Timer timer;

//   // Create the MDD
//   int32_t width{std::numeric_limits<int32_t>::max()};
//   width = 10;
//   MDD mdd(problem, width);

//   // Enforce all the constraints on the MDD
//   mdd.enforceConstraints(MDD::MDDConstructionAlgorithm::Separation);
//   std::cout << "Wallclock time enforce constraints (msec.): " <<
//           timer.getWallClockTimeMsec() << std::endl;

//   timer.reset();
//   timer.start();

//   double bestCost{std::numeric_limits<double>::max()};
//   mdd.dfsRec(mdd.getMDD(), bestCost, 0.0, mdd.getMDD());
//   std::cout << "Best solution: " << bestCost << std::endl;
//   mdd.printMDD("mdd");
// /*
//   // Create the MDD problem
//   //auto problem = std::make_shared<MDDProblem>();
//   problem->setMinimization();

//   // Add the list of variables
//   problem->addVariable(std::make_shared<Variable>(0, 0, 0, 0));
//   problem->addVariable(std::make_shared<Variable>(1, 1, 2, 5));
//   problem->addVariable(std::make_shared<Variable>(2, 2, 2, 5));
//   problem->addVariable(std::make_shared<Variable>(3, 3, 2, 5));
//   problem->addVariable(std::make_shared<Variable>(4, 4, 2, 5));
//   problem->addVariable(std::make_shared<Variable>(5, 5, 1, 1));
//   std::vector<int64_t> pickupNode{2, 4};
//   std::vector<int64_t> deliveryNode{3, 5};
//   std::vector<std::vector<int64_t>> costMatrix{
//     {0,    0,  389,  792, 1357,  961},
//     {0,    0,    0,    0,    0,    0},
//     {389,    0,    0,  641, 1226, 1168},
//     {792,    0,  641,    0, 1443, 1490},
//     {1357,    0, 1226, 1443,    0,  741},
//     {961,    0, 1168, 1490,  741,    0},
//   };

//   // Create the constraint
//   auto tsppd = std::make_shared<TSPPD>(pickupNode, deliveryNode, costMatrix);
//   tsppd->setScope(problem->getVariables());
//   problem->addConstraint(tsppd);

//   // Create the MDD
//   int32_t width{std::numeric_limits<int32_t>::max()};
//   width = 10;
//   MDD mdd(problem, width);

//   tools::Timer timer;

//   // Enforce all the constraints on the MDD
//   mdd.enforceConstraints(MDD::MDDConstructionAlgorithm::Separation);
//   std::cout << "Wallclock time enforce constraints (msec.): " <<
//           timer.getWallClockTimeMsec() << std::endl;

//   timer.reset();
//   timer.start();

//   std::vector<Edge*> solution;
//   if (problem->isMaximization())
//   {
//     solution = mdd.maximize();
//   }
//   else
//   {
//     //solution = mdd.minimize();
//     //mdd.dfs();
//     double bestCost{std::numeric_limits<double>::max()};
//     mdd.dfsRec(mdd.getMDD(), bestCost, 0.0, mdd.getMDD());
//     std::cout << "Best solution: " << bestCost << std::endl;
//   }

//   for (int idx = 0; idx < solution.size(); idx++)
//   {
//       auto edge = solution[idx];
//       std::cout << edge->getTail()->getLayer() << " - " << edge->getHead()->getLayer() << ": " <<
//               edge->getValue() << std::endl;
//   }
//   std::cout << "Wallclock time solution (msec.): " <<
//           timer.getWallClockTimeMsec() << std::endl;
//   mdd.printMDD("mdd");
//   */
// }


void runMDDOpt()
{
  using namespace mdd;

  // Create the MDD problem
  auto problem = std::make_shared<MDDProblem>();

  // Add the list of variables
  int32_t maxVars{3};
  for (int idx{0}; idx < maxVars; ++idx)
  {
    problem->addVariable(std::make_shared<Variable>(idx, idx, 4, 5));
  }

  // Create the constraint
  auto allDiff = std::make_shared<AllDifferent>();
  allDiff->setScope(problem->getVariables());
  //problem->addConstraint(allDiff);

  auto among = std::make_shared<Among>();
  among->setScope(problem->getVariables());
  among->setParameters({5}, 1, 1);
  problem->addConstraint(among);

  // Create the MDD
  int32_t width{std::numeric_limits<int32_t>::max()};
  width = 10;
  MDD mdd(problem, width);

  tools::Timer timer;

  // Enforce all the constraints on the MDD
  mdd.enforceConstraints(MDD::MDDConstructionAlgorithm::Filtering);
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
  mdd.printMDD("mdd");
}

}  // namespace

int main(int argc, char* argv[]) {

  // Run the MDD
  try
  {
    // TODO Add entry point code here
    runMDDOpt();
    // runTSPPD();
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
