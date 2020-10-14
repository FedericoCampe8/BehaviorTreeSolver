#include "mdd_optimization/td_optimizer.hpp"

#include <fstream>    // for std::ofstream
#include <iostream>
#include <stdexcept>  // for std::exception

// #define STRICT_CHECKS

namespace mdd {

TDMDDOptimizer::TDMDDOptimizer(MDDProblem::SPtr problem)
: pProblem(problem),
  pTimer(std::make_shared<tools::Timer>(true))
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

  // Compile the initial MDD
  pCompiler->compileMDD(TDCompiler::CompilationMode::Restricted);

  // Get the incumbent, i.e., the best solution found so far.
  // Note: incumbent should be found with a min/max path visit.
  // Here it is done using DFS.
  // TODO switch to min-path on directed acyclic graph
  double bestCost{std::numeric_limits<double>::max()};
  auto MDDGraph = pCompiler->getMDDMutable();
  std::vector<int64_t> path;
  path.reserve(pProblem->getVariables().size());
  dfsRec(MDDGraph, MDDGraph->getNodeState(0, 0), path, bestCost, 0, 0.0);

  // Update the solution cost
  updateSolutionCost(bestCost);

  return;

  bool improveIncumbent{true};
  if (pNumSolutionsCtr >= pNumMaxSolutions)
  {
    improveIncumbent = false;
  }

  while(improveIncumbent)
  {
    if (pTimer->getWallClockTimeMsec() > timeoutMsec)
    {
      // Return on timeout
      std::cout << "Exit on timeout" << std::endl;
      break;
    }

    //std::cout << "QUEUE SIZE: " << pCompiler->getMDDMutable()->getNumStoredStates() << std::endl;

    // Rebuild the MDD using the sorted queued states
    if (false /*!pCompiler->rebuildMDDFromQueue()*/)
    {
      // Nothing to recompile, return asap
      std::cout << "No more nodes to explore, exit" << std::endl;
      break;
    }

    // Compile the MDD
    if (false /*!pCompiler->compileMDD()*/)
    {
      continue;
    }

    // Get the incumbent, i.e., the best solution found so far
    bestCost = std::numeric_limits<double>::max();

    // Get the best cost found which, in case of top-down restricted
    // is equivalent to the cumulative cost stored at the tail node
    //bestCost = mddGraph->getNodeState(mddGraph->getNumLayers(), 0)->cumulativeCost();

#ifdef STRICT_CHECKS
  double bestCostOnDFS{std::numeric_limits<double>::max()};
  dfsRec(mddGraph, bestCostOnDFS, 0, -1, 0.0);
  assert(bestCost == bestCostOnDFS);
#endif

    // Update the solution cost
    updateSolutionCost(bestCost);

    if (pNumSolutionsCtr >= pNumMaxSolutions)
    {
      std::cout << "Reached the maximum number of solutions, exit" << std::endl;
      break;
    }
  }  // while improvements
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

void TDMDDOptimizer::dfsRec(TopDownMDD* mddGraph, DPState* state, std::vector<int64_t>& path,
                            double& bestCost, const uint32_t currLayer, double cost)
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

    // Get the head of the edge and prepare next state
    auto head = mddGraph->getNodeState(currLayer, edge->head);
    for (auto val : edge->valuesList)
    {
      // Consider all values on a parallel edge
      const auto currentCost = state->getCostPerValue(val);

      // Update next state
      path.push_back(val);
      head->updateState(state, val);
      head->forceCumulativePath(path);

      // Call the recursion
      cost += currentCost;
      dfsRec(mddGraph, head, path, bestCost, currLayer + 1, cost);
      cost -= currentCost;

      path.pop_back();
    }
  }
}

void TDMDDOptimizer::printMDD(const std::string& outFileName)
{
  pCompiler->printMDD(outFileName);
}

}  // namespace mdd
