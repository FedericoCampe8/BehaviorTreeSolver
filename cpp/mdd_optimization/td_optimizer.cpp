#include "mdd_optimization/td_optimizer.hpp"

#include <fstream>    // for std::ofstream
#include <iostream>
#include <stdexcept>  // for std::exception

#define STRICT_CHECKS

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
  pCompiler->compileMDD();

  // Get the incumbent, i.e., the best solution found so far
  double bestCost{std::numeric_limits<double>::max()};

  // Get the best cost found which, in case of top-down restricted
  // is equivalent to the cumulative cost stored at the tail node
  auto mddGraph = pCompiler->getMDDMutable();
  bestCost = mddGraph->getNodeState(mddGraph->getNumLayers(), 0)->cumulativeCost();

#ifdef STRICT_CHECKS
  double bestCostOnDFS{std::numeric_limits<double>::max()};
  dfsRec(mddGraph, bestCostOnDFS, 0, -1, 0.0);
  assert(bestCost == bestCostOnDFS);
#endif

  // Update the solution cost
  updateSolutionCost(bestCost);

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

    // Rebuild the MDD using the sorted queued states
    if (!pCompiler->rebuildMDDFromQueue())
    {
      // Nothing to recompile, return asap
      std::cout << "No more nodes to explore, exit" << std::endl;
      break;
    }

    // Compile the MDD
    if (!pCompiler->compileMDD())
    {
      continue;
    }

    // Get the incumbent, i.e., the best solution found so far
    bestCost = std::numeric_limits<double>::max();

    // Get the best cost found which, in case of top-down restricted
    // is equivalent to the cumulative cost stored at the tail node
    bestCost = mddGraph->getNodeState(mddGraph->getNumLayers(), 0)->cumulativeCost();

#ifdef STRICT_CHECKS
  double bestCostOnDFS{std::numeric_limits<double>::max()};
  dfsRec(mddGraph, bestCostOnDFS, 0, -1, 0.0);
  assert(bestCost == bestCostOnDFS);
#endif

    // Update the solution cost
    updateSolutionCost(bestCost);

    if (pNumSolutionsCtr >= pNumMaxSolutions)
    {
      break;
    }
  }  // while improvements
}

void TDMDDOptimizer::updateSolutionCost(double cost)
{
  const auto oldBestCostForPrint = pBestCost;
  if (cost <= pBestCost)
  {
    ++pNumSolutionsCtr;
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

void TDMDDOptimizer::dfsRec(TopDownMDD* mddGraph, double& bestCost, const uint32_t currLayer,
                            const int32_t currHead, double cost)
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

  auto activeEdgeList = mddGraph->getActiveEdgesOnLayer(currLayer);
  for (auto edge : activeEdgeList)
  {
    if ((currHead >= 0) && (edge->tail != currHead))
    {
      continue;
    }
    auto tailState = mddGraph->getNodeState(currLayer, edge->tail);
    cost += tailState->getCostPerValue(edge->value);
    dfsRec(mddGraph, bestCost, currLayer + 1, static_cast<int32_t>(edge->head), cost);
    cost -= tailState->getCostPerValue(edge->value);
  }
}

void TDMDDOptimizer::printMDD(const std::string& outFileName)
{
  pCompiler->printMDD(outFileName);
}

}  // namespace mdd
