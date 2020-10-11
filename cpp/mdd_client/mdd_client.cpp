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
#include <sparsepp/spp.h>

#include "mdd_optimization/all_different.hpp"
#include "mdd_optimization/among.hpp"
#include "mdd_optimization/job_shop.hpp"
#include "mdd_optimization/mdd.hpp"
#include "mdd_optimization/mdd_optimizer.hpp"
#include "mdd_optimization/mdd_problem.hpp"
#include "mdd_optimization/td_compiler.hpp"
#include "mdd_optimization/td_optimizer.hpp"
#include "mdd_optimization/top_down_mdd.hpp"
#include "mdd_optimization/tsppd.hpp"
#include "mdd_optimization/variable.hpp"
#include "tools/timer.hpp"

namespace {

void runJobShop()
{
  using namespace mdd;

  const std::string resPath =  "../cpp/mdd_client/data_job_shop/res.json";
  const std::string taskPath = "../cpp/mdd_client/data_job_shop/task.json";

  std::ifstream resfile(resPath);
  std::string resString((std::istreambuf_iterator<char>(resfile)),
                        (std::istreambuf_iterator<char>()));

  std::ifstream taskfile(taskPath);
  std::string taskString((std::istreambuf_iterator<char>(taskfile)),
                         (std::istreambuf_iterator<char>()));

  // Parse the inputs into Json
  rapidjson::Document resDoc;
  resDoc.Parse(resString.c_str());

  rapidjson::Document taskDoc;
  taskDoc.Parse(taskString.c_str());

  // Parse the resources
  const auto& resJson = resDoc["res"];
  uint64_t numMachines = 0;
  for (rapidjson::SizeType idx{0}; idx < resJson.Size(); ++idx)
  {
    const auto& resObj = resJson[idx].GetObject();
    const auto machineId = resObj.FindMember("id")->value.GetInt();
    if (numMachines < machineId)
    {
      numMachines = machineId;
    }

    std::vector<std::pair<int64_t, int64_t>> availability;
    const auto& machineAvailability = resObj.FindMember("periods")->value.GetArray();
    for (rapidjson::SizeType availIdx{0}; availIdx < machineAvailability.Size(); ++availIdx)
    {
      const auto& availObj = machineAvailability[availIdx].GetObject();
      const auto start = availObj.FindMember("s")->value.GetInt() * 10;
      const auto end = availObj.FindMember("e")->value.GetInt() * 10;
      availability.push_back({start, end});
    }
  }
  // Identifiers starts from zero
  ++numMachines;

  // Parse the tasks
  JobShopState::TaskSpecMap taskSpecMap;
  const auto& taskJson = taskDoc["tasks"];
  for (rapidjson::SizeType idx{0}; idx < taskJson.Size(); ++idx)
  {
    const auto& taskObj = taskJson[idx].GetObject();

    const auto jobId = taskObj.FindMember("id")->value.GetInt();
    const auto jobDuration = static_cast<int>(taskObj.FindMember("dur")->value.GetDouble() * 10);
    const auto jobDependencyList = taskObj.FindMember("dep")->value.GetArray();
    const auto jobResourcesList = taskObj.FindMember("res")->value.GetArray();
    const auto resNeeded = jobResourcesList[0].GetInt();


    // Note: 1 job <-> 1 task
    taskSpecMap[jobId].push_back(resNeeded);
    taskSpecMap[jobId].push_back(jobDuration);
    taskSpecMap[jobId].push_back(10);
    for (rapidjson::Value::ConstValueIterator itr = jobDependencyList.Begin();
            itr != jobDependencyList.End(); ++itr)
    {
      const auto depId = itr->GetInt();
      if (depId > -1)
      {
        taskSpecMap[jobId].push_back(depId);
      }
    }
  }

  // Create the MDD problem
  auto problem = std::make_shared<MDDProblem>();
  problem->setMinimization();

  // Create a variable per task
  // First variable corresponds to the first task that doesn't require
  // any dependencies and it can be scheduled right away
  problem->addVariable(std::make_shared<Variable>(0, 0, 0, 0));
  const auto numTasks = static_cast<int>(taskSpecMap.size());
  for (int taskIdx{1}; taskIdx < numTasks; ++taskIdx)
  {
    problem->addVariable(std::make_shared<Variable>(taskIdx, taskIdx, 1, numTasks-1));
  }

  std::cout << "Num. Machines: " << numMachines << std::endl;
  std::cout << "Num. Tasks: " << numTasks << std::endl;

  auto jobShopConstraint = std::make_shared<JobShop>(taskSpecMap, numMachines);
  jobShopConstraint->setScope(problem->getVariables());
  problem->addConstraint(jobShopConstraint);

  // Create the optimizer
  MDDOptimizer optimizer(problem);
  int32_t width{2};

  // Run optimization
  tools::Timer timer;
  uint64_t timeoutMsec{40000};
  optimizer.runOptimization(width, timeoutMsec);
  std::cout << "Wallclock time Branch and Bound optimization (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;

  optimizer.printMDD("mdd");
}

void runTSPPD()
{
  using namespace mdd;

  std::string instancePath{"../cpp/mdd_client/data/grubhub-15-9.json"};
  std::ifstream datafile(instancePath);
  std::string dataString((std::istreambuf_iterator<char>(datafile)),
                         (std::istreambuf_iterator<char>()));

  // Parse the inputs into Json
  rapidjson::Document dataDoc;
  dataDoc.Parse(dataString.c_str());

  // Parse the nodes
  const auto& nodesJson = dataDoc["nodes"];
  int numVars = static_cast<int>(nodesJson.Size());

  // Parse the distance matrix
  const auto& edgesJson = dataDoc["edges"];
  std::vector<std::vector<int64_t>> costMatrix;
  costMatrix.resize(edgesJson.Size());
  for (rapidjson::SizeType idx{0}; idx < edgesJson.Size(); ++idx)
  {
    const auto& resObj = edgesJson[idx].GetArray();
    auto& row = costMatrix[idx];
    row.resize(resObj.Size());
    for (rapidjson::SizeType idx2{0}; idx2 < resObj.Size(); ++idx2)
    {
      row[idx2] = resObj[idx2].GetInt64();
    }
  }

  // Create the MDD problem
  auto problem = std::make_shared<MDDProblem>();
  problem->setMinimization();

  /*
  /////// ALL DIFFERENT ///////
  numVars = 5;
  for (int idx{0}; idx < numVars; ++idx)
  {
    problem->addVariable(std::make_shared<Variable>(idx, idx, 1, numVars));
  }

  spp::sparse_hash_set<int64_t> allDiffVals;
  for (int64_t val{1}; val <= numVars; ++val)
  {
    allDiffVals.insert(val);
  }
  auto allDiff = std::make_shared<AllDifferent>(allDiffVals);
  allDiff->setScope(problem->getVariables());
  problem->addConstraint(allDiff);

  tools::Timer timerAllDiff;
  int32_t widthAllDiff{2};
  TDMDDOptimizer allDiffOptimizer(problem);
  allDiffOptimizer.runOptimization(widthAllDiff, 60000);
  allDiffOptimizer.setMaxNumSolutions(3);
  std::cout << "Wallclock time build (msec.): " << timerAllDiff.getWallClockTimeMsec() << std::endl;
  allDiffOptimizer.printMDD("topdown_mdd");
  return;
  /////////////////////////////
   */




  // First variable is always the node "+0"
  problem->addVariable(std::make_shared<Variable>(0, 0, 0, 0));

  spp::sparse_hash_map<int64_t, int64_t> pickupDeliveryMap;
  pickupDeliveryMap[0] = 1;
  for (int idx{1}; idx < numVars-1; ++idx)
  {
    if (((idx+1) % 2) == 0)
    {
      pickupDeliveryMap[idx + 1] = idx + 2;
    }

    problem->addVariable(std::make_shared<Variable>(idx, idx, 2, numVars-1));
  }

  /*
  std::cout << "Pickup and delivery locations: " << std::endl;
  for (const auto& locIter : pickupDeliveryMap)
  {
    std::cout << locIter.first << " -> " << locIter.second << "\n";
  }
  */
  std::cout << "Num Vars: " << numVars << " with domain [" << 2 << ", " << numVars - 1 << "]\n";

  // Last variable is always the node "-0"
  problem->addVariable(std::make_shared<Variable>(numVars-1, numVars-1, 1, 1));

  // Create the constraint
  auto tsppd = std::make_shared<TSPPD>(pickupDeliveryMap, costMatrix);
  tsppd->setScope(problem->getVariables());
  problem->addConstraint(tsppd);

  tools::Timer timerMDD;
  int32_t width{2};
  uint64_t timeoutMsec{60000};
  TDMDDOptimizer tdOptimizer(problem);
  tdOptimizer.runOptimization(width, timeoutMsec);
  tdOptimizer.setMaxNumSolutions(10000000);
  //TDCompiler tdCompiler(problem, width);
  //tdCompiler.compileMDD();
  std::cout << "Wallclock time build (msec.): " << timerMDD.getWallClockTimeMsec() << std::endl;
  std::cout << "Number of solutions found " << tdOptimizer.getNumSolutions() << std::endl;
  tdOptimizer.printMDD("topdown_mdd");
  return;

  // Create the optimizer
  MDDOptimizer optimizer(problem);

  // Run optimization
  tools::Timer timer;
  uint64_t timeoutMsecOpt{40000};
  optimizer.runOptimization(width, timeoutMsecOpt);
  std::cout << "Wallclock time Branch and Bound optimization (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;

  optimizer.printMDD("mdd");

  // Create the MDD
  /*
  for (int i = 0; i < 10000; ++i)
  {
    MDD mdd(problem, width);

    // Enforce all the constraints on the MDD
    mdd.enforceConstraints(MDD::MDDConstructionAlgorithm::RestrictedTopDown);
    //std::cout << "Wallclock time enforce constraints (msec.): " <<
    //        timer.getWallClockTimeMsec() << std::endl;

    timer.reset();
    timer.start();

    double oldCost = std::numeric_limits<double>::max();
    mdd.dfsRec(mdd.getMDD(), oldCost, numVars, 0.0, mdd.getMDD());
    if (oldCost < bestCost)
    {
      bestCost = oldCost;
      std::cout << "Improving solution: " << bestCost << std::endl;
    }
  }
  std::cout << "Best solution: " << bestCost << std::endl;
  */

  /*
  double bestCost{std::numeric_limits<double>::max()};
  int32_t width{std::numeric_limits<int32_t>::max()};
  width = 3;
  MDD mdd(problem, width);

  // Enforce all the constraints on the MDD
  mdd.enforceConstraints(MDD::MDDConstructionAlgorithm::RestrictedTopDown);
  std::cout << "Wallclock time enforce constraints (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;

  timer.reset();
  timer.start();
  mdd.dfsRec(mdd.getMDD(), bestCost, numVars, 0.0, mdd.getMDD());
  std::cout << "Best solution: " << bestCost << std::endl;
  mdd.printMDD("mdd");
  */

  /*
  // Create the MDD problem
  auto problem = std::make_shared<MDDProblem>();
  problem->setMinimization();

  // Add the list of variables
  problem->addVariable(std::make_shared<Variable>(0, 0, 0, 0));
  problem->addVariable(std::make_shared<Variable>(1, 1, 2, 5));
  problem->addVariable(std::make_shared<Variable>(2, 2, 2, 5));
  problem->addVariable(std::make_shared<Variable>(3, 3, 2, 5));
  problem->addVariable(std::make_shared<Variable>(4, 4, 2, 5));
  problem->addVariable(std::make_shared<Variable>(5, 5, 1, 1));
  std::vector<int64_t> pickupNode{2, 4};
  std::vector<int64_t> deliveryNode{3, 5};
  std::vector<std::vector<int64_t>> costMatrix{
    {0,    0,  389,  792, 1357,  961},
    {0,    0,    0,    0,    0,    0},
    {389,    0,    0,  641, 1226, 1168},
    {792,    0,  641,    0, 1443, 1490},
    {1357,    0, 1226, 1443,    0,  741},
    {961,    0, 1168, 1490,  741,    0},
  };

  // Create the constraint
  auto tsppd = std::make_shared<TSPPD>(pickupNode, deliveryNode, costMatrix);
  tsppd->setScope(problem->getVariables());
  problem->addConstraint(tsppd);

  // Create the MDD
  int32_t width{std::numeric_limits<int32_t>::max()};
  width = 10;
  MDD mdd(problem, width);

  tools::Timer timer;

  // Enforce all the constraints on the MDD
  mdd.enforceConstraints(MDD::MDDConstructionAlgorithm::TopDown);
  std::cout << "Wallclock time enforce constraints (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;

  timer.reset();
  timer.start();

  std::vector<Edge*> solution;
  if (problem->isMaximization())
  {
    solution = mdd.maximize();
  }
  else
  {
    //solution = mdd.minimize();
    //mdd.dfs();
    double bestCost{std::numeric_limits<double>::max()};
    mdd.dfsRec(mdd.getMDD(), bestCost, 0.0, mdd.getMDD());
    std::cout << "Best solution: " << bestCost << std::endl;
  }

  for (int idx = 0; idx < solution.size(); idx++)
  {
      auto edge = solution[idx];
      std::cout << edge->getTail()->getLayer() << " - " << edge->getHead()->getLayer() << ": " <<
              edge->getValue() << std::endl;
  }
  std::cout << "Wallclock time solution (msec.): " <<
          timer.getWallClockTimeMsec() << std::endl;
  mdd.printMDD("mdd");
*/
}

/*
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
*/
}  // namespace

int main(int argc, char* argv[]) {

  // Run the MDD
  try
  {
    // TODO Add entry point code here
    //runMDDOpt();
    if (argc > 1)
    {
      std::cout << "RUN JOB-SHOP\n";
      runJobShop();
    }
    else
    {
      std::cout << "RUN TSPPD\n";
      runTSPPD();
    }
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
