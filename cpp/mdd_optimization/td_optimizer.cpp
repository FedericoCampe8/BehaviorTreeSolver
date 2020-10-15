#include "mdd_optimization/td_optimizer.hpp"

#include <cassert>
#include <fstream>    // for std::ofstream
#include <iostream>
#include <stdexcept>  // for std::exception
#include <utility>    // for std::move

// #define DEBUG

namespace mdd {

TDMDDOptimizer::TDMDDOptimizer(MDDProblem::SPtr problem)
: pProblem(problem)
{
  if (pProblem == nullptr)
  {
    throw std::invalid_argument("TDMDDOptimizer - empty pointer to the problem");
  }

  if (pProblem->isMaximization())
  {
    throw std::invalid_argument("TDMDDOptimizer - "
            "MDDOptimizer only supports minimization problems");
  }
}

void TDMDDOptimizer::runOptimization(uint32_t width, uint64_t timeoutMsec)
{
  pMaxWidth = width;

  // Use a top-down compiler for B&B optimization
  pCompiler = std::make_unique<TDCompiler>(pProblem, pMaxWidth);

  // Get the pointer to the MDD graph data structure
  auto mddGraph = pCompiler->getMDDMutable();

  // Initialize the queue of nodes to branch on
  // with the root node
  pQueue.push_back(DPState::UPtr(mddGraph->getNodeState(0, 0)->clone()));

  // Run branch and bound optimization problem
  runBranchAndBound(timeoutMsec);
}

void TDMDDOptimizer::updateSolutionCost(double cost)
{
  // Increase the total number of solutions found
  ++pNumSolutionsCtr;

  const auto oldBestCostForPrint = pBestCost;
  if (cost <= pBestCost)
  {
    pBestCost = cost;

    // If more solutions are needed, allow to find solutions with the same cost
    auto newIncumbent = pBestCost;
    if (pNumSolutionsCtr < pNumMaxSolutions)
    {
      newIncumbent += 0.1;
    }
    pCompiler->setIncumbent(newIncumbent);

    if (cost < oldBestCostForPrint)
    {
      std::cout << "New incumbent: " << pBestCost << " at " <<
              pTimer->getWallClockTimeMsec() << " msec" <<  std::endl;
    }
  }
}

void TDMDDOptimizer::updateSolutionLowerBound(double cost)
{
  if (cost <= pBestCost && cost > pAdmissibleLowerBound)
  {
    pAdmissibleLowerBound = cost;
    std::cout << "New lower bound: " << pAdmissibleLowerBound << " at " <<
            pTimer->getWallClockTimeMsec() << " msec" <<  std::endl;
  }
}

void TDMDDOptimizer::runBranchAndBound(uint64_t timeoutMsec)
{
  // Start the timer
  pTimer = std::make_shared<tools::Timer>();

  // Get the pointer to the MDD
  auto mdd = pCompiler->getMDDMutable();

  // Start the optimization process on the queue
  std::vector<int64_t> path;
  path.reserve(pProblem->getVariables().size());
  double bestCost{std::numeric_limits<double>::max()};
  while(!pQueue.empty())
  {
    // Return on timeout
    if (pTimer->getWallClockTimeMsec() > timeoutMsec)
    {
      // Return on timeout
      std::cout << "Exit on timeout at (msec.) " << pTimer->getWallClockTimeMsec() << std::endl;
      break;
    }

    // Choose a node to branch on
    auto node = selectNodeForBranching();
    assert(node != nullptr);

    // Obtain an incumbent by building a restricted MDD starting from "node"
    // Keep a copy of the node for future possible branching

#ifdef DEBUG
    tools::Timer timerRestricted;
    std::cout << "Build restricted MDD\n";
#endif
    pCompiler->compileMDD(TDCompiler::CompilationMode::Restricted, DPState::UPtr(node->clone()));
#ifdef DEBUG
    std::cout << "Done in (msec): " << timerRestricted.getWallClockTimeMsec() << std::endl;
#endif

    // Get the incumbent, i.e., the best solution found so far.
    // Note: incumbent should be found with a min/max path visit.
    // Here it is done using DFS.
    // TODO switch to min-path on directed acyclic graph
    path.clear();
    bestCost = std::numeric_limits<double>::max();
    dfsRec(mdd, mdd->getNodeState(0, 0), path, bestCost, 0, 0.0);

    // Update the solution cost
    updateSolutionCost(bestCost);

    // Check if the MDD is exact, if not proceed with branch and bound
    if (!pCompiler->isExact())
    {
      // Branch on the node and obtain a lower bound by building a relaxed MDD
#ifdef DEBUG
      tools::Timer timerRelaxed;
      std::cout << "Build relaxed MDD\n";
#endif
      pCompiler->compileMDD(TDCompiler::CompilationMode::Relaxed, std::move(node));
#ifdef DEBUG
      std::cout << "Done in (msec): " << timerRelaxed.getWallClockTimeMsec() << std::endl;
#endif

      // Get the lower bound
      path.clear();
      bestCost = std::numeric_limits<double>::max();
      dfsRec(mdd, mdd->getNodeState(0, 0), path, bestCost, 0, 0.0, true);
      updateSolutionLowerBound(bestCost);

      // Check if the percentage (delta) between lower bound and upper bound is acceptable
      if (pAdmissibleLowerBound < pBestCost)
      {
        const auto diffBounds = pBestCost - pAdmissibleLowerBound;
        const auto optGap = (diffBounds / pBestCost) * 100.0;
        if (optGap <= pDeltaOnSolution)
        {
          // Return on delta between lower and upper bound
          std::cout << "Exit on optimality gap of " << optGap <<
                  " at (msec.) " << pTimer->getWallClockTimeMsec() << std::endl;
          break;
        }
      }

      // If it is not possible to prune the search using this bound,
      // identify and exact cutset of the MDD and add the nodes of the cutset
      // into the queue
      if (bestCost < pBestCost)
      {
        processCutset();
      }
    }
  }
}

DPState::UPtr TDMDDOptimizer::selectNodeForBranching()
{
  int bestIdx{-1};
  double bestCost{std::numeric_limits<double>::max()};
  for (int idx{0}; idx < static_cast<int>(pQueue.size()); ++idx)
  {
    if (pQueue.at(idx)->cumulativeCost() < bestCost)
    {
      bestCost = pQueue.at(idx)->cumulativeCost();
      bestIdx = idx;
    }
  }

  if (bestIdx >= 0)
  {
    auto bestState = std::move(pQueue[bestIdx]);
    pQueue.erase(pQueue.begin() + bestIdx);
    return std::move(bestState);
  }
  else
  {
    return nullptr;
  }
}

void TDMDDOptimizer::processCutset()
{
  // The frontier cutset is computed as follows:
  //   for each w in 1 <= w <= max_width:
  //     q = selectLowestExactNode on column(w);
  //     queue <- {q}
  // Get the pointer to the MDD
  auto mdd = pCompiler->getMDDMutable();
  std::vector<DPState*> frontier;
  for (uint32_t idx{0}; idx < mdd->getMaxWidth(); ++idx)
  {
    // For each vertical layer traverse top down to find the lowest exact node
    DPState* exactNode{nullptr};
    for (uint32_t lidx{1}; lidx < mdd->getNumLayers(); ++lidx)
    {
      if (mdd->isReachable(lidx, idx) && mdd->getNodeState(lidx+1, idx)->isExact())
      {
        exactNode = mdd->getNodeState(lidx+1, idx);
      }
      else
      {
        break;
      }
    }

    if (exactNode != nullptr)
    {
      frontier.push_back(exactNode);
#ifdef DEBUG
      std::cout << "EXACT NODE: " << std::endl;
      std::cout << exactNode->toString() << std::endl;
#endif
    }
  }

#ifdef DEBUG
  if (frontier.size() < mdd->getMaxWidth())
  {
    std::cout << "Incomplete frontier of size " << frontier.size() << std::endl;
    getchar();
  }
#endif

  // Set the nodes of the frontier in the queue
  for (auto node : frontier)
  {
    pQueue.push_back(DPState::UPtr(node->clone()));
  }
}


void TDMDDOptimizer::dfsRec(TopDownMDD* mddGraph, DPState* state, std::vector<int64_t>& path,
                            double& bestCost, const uint32_t currLayer, double cost, bool isRelaxed)
{
  if (currLayer == mddGraph->getNumLayers())
  {
    // Get the cost on the tail node
    if (cost < bestCost)
    {
      bestCost = cost;
    }
    return;
  }

  for (auto edge : mddGraph->getActiveEdgesOnLayer(currLayer))
  {
    if ((currLayer > 0) && (mddGraph->getNodeState(currLayer-1, edge->tail) != state))
    {
      // Skip nodes that are not from the same parent
      continue;
    }

    if (isRelaxed && bestCost < std::numeric_limits<double>::max())
    {
      continue;
    }

    // Get the head of the edge and prepare next state
    auto head = mddGraph->getNodeState(currLayer, edge->head);
    for (auto val : edge->valuesList)
    {
      // Consider all values on a parallel edge
      const auto currentCost = state->getCostPerValue(val);
      if (currentCost > bestCost)
      {
        // break on higher cost than the best cost found so far
        continue;
      }

      // Update next state
      path.push_back(val);
      head->updateState(state, val);
      head->forceCumulativePath(path);

      // Call the recursion
      cost += currentCost;
      dfsRec(mddGraph, head, path, bestCost, currLayer + 1, cost, isRelaxed);
      cost -= currentCost;

      path.pop_back();

      // Consider only the first value
      break;
    }
  }
}

void TDMDDOptimizer::printMDD(const std::string& outFileName)
{
  pCompiler->printMDD(outFileName);
}

}  // namespace mdd