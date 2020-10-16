#include "mdd_optimization/td_optimizer.hpp"

#include <cassert>
#include <fstream>    // for std::ofstream
#include <iostream>
#include <stdexcept>  // for std::exception
#include <utility>    // for std::move

//#define DEBUG

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
    /*
    std::cout << "New lower bound: " << pAdmissibleLowerBound << " at " <<
            pTimer->getWallClockTimeMsec() << " msec" <<  std::endl;
    */
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
    const auto builtRestricted = pCompiler->compileMDD(
            TDCompiler::CompilationMode::Restricted, DPState::UPtr(node->clone()));

    // Get the incumbent, i.e., the best solution found so far.
    // Note: incumbent should be found with a min/max path visit.
    // Here it is done using DFS.
    // TODO switch to min-path on directed acyclic graph
    /*
    if (builtRestricted)
    {
      std::cout << "PRINT RESTRICTED\n";
      printMDD("restricted_mdd");
      getchar();
    }
    */
    path.clear();
    bestCost = std::numeric_limits<double>::max();
    dfsRec(mdd, mdd->getNodeState(0, 0), path, bestCost, 0, false);

    // Update the solution cost
    updateSolutionCost(bestCost);

    // Check if the MDD is exact, if not proceed with branch and bound
    if (!pCompiler->isExact() || !builtRestricted)
    {
      // Branch on the node and obtain a lower bound by building a relaxed MDD
      const auto builtRelaxed = pCompiler->compileMDD(
              TDCompiler::CompilationMode::Relaxed, std::move(node));

      if (!builtRelaxed)
      {
        // The built was not successful meaning that the compiler couldn't find
        // an MDD that led to a cost lower than the incumbent.
        // Continue with next node in the queue
        continue;
      }

      // Get the lower bound
      path.clear();
      bestCost = std::numeric_limits<double>::max();

      /*
      if (builtRelaxed)
      {
        std::cout << "PRINT RELAXED\n";
        printMDD("relaxed_mdd");
        getchar();
      }
      */
      dfsRec(mdd, mdd->getNodeState(0, 0), path, bestCost, 0, true);
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
      // identify an exact cutset of the MDD and add the nodes of the cutset
      // into the queue
      if (bestCost < pBestCost)
      {
        processCutset();
      }
    }
    else
    {
      std::cout << "Exit exact restricted MDD at (msec.) " <<
              pTimer->getWallClockTimeMsec() << std::endl;
    }
  }

  if (pQueue.empty())
  {
    std::cout << "Exit on empty queue at (msec.) " <<
            pTimer->getWallClockTimeMsec() << std::endl;
  }
}

DPState::UPtr TDMDDOptimizer::selectNodeForBranching()
{
  int bestIdx{-1};
  double bestCost{std::numeric_limits<double>::max()};
  for (int idx{0}; idx < static_cast<int>(pQueue.size()); ++idx)
  {
    const auto costPath = pProblem->getConstraints().at(0)->calculateCost(
            pQueue.at(idx)->cumulativePath());
    if (costPath >= pBestCost)
    {
      pQueue.erase(pQueue.begin() + idx);
      idx = idx - 1;
      continue;
    }

    if (pQueue.at(idx)->cumulativeCost() < bestCost)
    {
      bestCost = pQueue.at(idx)->cumulativeCost();
      bestIdx = idx;
    }
  }

  DPState::UPtr outNode{nullptr};
  if (bestIdx >= 0)
  {
    outNode = std::move(pQueue[bestIdx]);
    pQueue.erase(pQueue.begin() + bestIdx);
  }
  assert(pProblem->getConstraints().at(0)->calculateCost(outNode->cumulativePath()) < pBestCost);

  return std::move(outNode);
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
  std::vector<uint32_t> edgeLayerFrontier;
  frontier.resize(mdd->getMaxWidth(), nullptr);
  edgeLayerFrontier.resize(mdd->getMaxWidth());


  uint32_t currLayer{0};
  bool findNodes{true};
  frontier[0] = mdd->getNodeState(0, 0);
  while (findNodes && (currLayer < mdd->getNumLayers() - 1))
  {
    for (uint32_t w{0}; w < mdd->getMaxWidth(); ++w)
    {
      auto node = frontier[w];
      if (node == nullptr)
      {
        continue;
      }

      for (auto edge : mdd->getActiveEdgesOnLayerGivenTail(currLayer, w))
      {
        auto head = edge->head;
        auto newNodePtr = mdd->getNodeState(currLayer+1, head);
        frontier[head] = newNodePtr;
        edgeLayerFrontier[head] = currLayer;
      }
    }

    // Try next layer
    ++currLayer;

    // Try to terminate the loop
    findNodes = false;
    for (auto node : frontier)
    {
      if (node == nullptr)
      {
        // Continue looping to add nodes
        findNodes = true;
        break;
      }
    }
  }

  for (uint32_t w{0}; w < mdd->getMaxWidth(); ++w)
  {
    auto node = frontier[w];
    if (node->isExact())
    {
      pQueue.push_back(DPState::UPtr(node->clone()));
    }
    else
    {
      // Split the node on is incoming edges
      auto layerForNotExactNode = edgeLayerFrontier[w];
      for (auto edge : mdd->getActiveEdgesOnLayer(layerForNotExactNode))
      {
        if (edge->head == w)
        {
          // This edge is active and it is pointing to the non exact node
          // Create a new node for each value on the edge
          auto tailNode = mdd->getNodeState(layerForNotExactNode, edge->tail);
          assert(tailNode->isExact());
          for (auto val : edge->valuesList)
          {
            auto clonedNode = tailNode->clone();
            clonedNode->updateState(clonedNode, val);
            clonedNode->setExact(true);
            clonedNode->setNonDefaultState();

            // Store the node in the queue
            pQueue.push_back(DPState::UPtr(clonedNode));
          }
        }
      }
    }
  }
}

double TDMDDOptimizer::calculateMinPath()
{
  // Get the graph
  auto mdd = pCompiler->getMDDMutable();

  // Initialize the cost list
  std::vector<double> nodeCostList;

  // There is one root, one tail node and (num. layers - 1) * width nodes in the MDD
  const auto width = mdd->getMaxWidth();
  const auto numLayers = mdd->getNumLayers();
  const auto numNodes = ((numLayers - 1) * width) + 2;
  nodeCostList.resize(numNodes, std::numeric_limits<double>::max());

  // Start from the root with cost zero
  nodeCostList[0] = 0.0;

  // Consider the nodes in a topological order.
  // Note: the topological order is encapsulated by the structure of the MDD:
  // top-down, left-to-right.
  // The algorithm works like follows:
  //  for each node u in topological order:
  //    for each adjacent vertex v of u:
  //      if (dist[v] > dist[u] + weight(u, v)) then
  //        dist[v] = dist[u] + weight(u, v)
  // Note: all the arcs in the graph must have their cost initialized
  // before starting the min-path algorithm.
  for (uint32_t lidx{0}; lidx < numLayers; ++lidx)
  {
    for (uint32_t nidx{0}; nidx < width; ++nidx)
    {
      if (lidx == 0 && nidx > 0)
      {
        // Break on layer 0 after the first node since layer 0
        // contains only the root
        break;
      }

      // Update all nodes reachable from the current node with their cost
      // considering the edges connecting them
      for (auto edge : mdd->getActiveEdgesOnLayerGivenTail(lidx, nidx))
      {
        // Get the index of the corresponding reachable node
        assert(edge->tail == nidx);
        const auto tailNodeIdx = (lidx == 0) ? 0 : ((lidx - 1) * width + edge->tail + 1);
        const auto headNodeIdx = (lidx - 1) * width + edge->head + 1;
        for (auto costVal : edge->costList)
        {
          if (nodeCostList.at(tailNodeIdx) + costVal < nodeCostList.at(headNodeIdx))
          {
            nodeCostList[headNodeIdx] = nodeCostList.at(tailNodeIdx) + costVal;
          }
        }
      }
    }
  }
  return nodeCostList.back();
}

void TDMDDOptimizer::dfsRec(TopDownMDD* mddGraph, DPState* state, std::vector<int64_t>& path,
                            double& bestCost, const uint32_t currLayer, bool isRelaxed)
{
  if (currLayer == mddGraph->getNumLayers())
  {
    // Get the cost on the tail node
    const auto costPath = pProblem->getConstraints().at(0)->calculateCost(path);
    if (costPath < bestCost)
    {
      bestCost = costPath;
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

    if (false && isRelaxed && bestCost < std::numeric_limits<double>::max())
    {
      continue;
    }

    // Get the head of the edge and prepare next state
    auto head = mddGraph->getNodeState(currLayer, edge->head);
    for (auto val : edge->valuesList)
    {
      // Call the recursion
      path.push_back(val);
      dfsRec(mddGraph, head, path, bestCost, currLayer + 1, isRelaxed);
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
