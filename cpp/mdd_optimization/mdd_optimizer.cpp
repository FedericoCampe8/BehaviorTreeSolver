#include "mdd_optimization/mdd_optimizer.hpp"

#include <fstream>    // for std::ofstream
#include <iostream>
#include <stdexcept>  // for std::exception

#include "mdd_optimization/top_down_compiler.hpp"
#include "tools/timer.hpp"

namespace {
constexpr uint32_t kLayerZero{0};
}

namespace mdd {

MDDOptimizer::MDDOptimizer(MDDProblem::SPtr problem)
: pProblem(problem),
  pArena(std::make_unique<Arena>())
{
  if (pProblem == nullptr)
  {
    throw std::invalid_argument("MDD - empty pointer to the problem");
  }

  if (pProblem->isMaximization())
  {
    throw std::invalid_argument("MDD - MDDOptimizer only supports minimization problems");
  }

  // Resize the number of layers of this MDD to have one layer per variable in the problem.
  // Note: there is one more corresponding to the terminal node layer
  pMDDGraph.resize(pProblem->getVariables().size() + 1);
}

void MDDOptimizer::runOptimization(int32_t width, uint64_t timeoutMsec)
{
  if (width < 1)
  {
    throw std::invalid_argument("MDD - runOptimization: invalid width size");
  }
  pMaxWidth = width;

  // Use a top-down compiler for B&B optimization
  pMDDCompiler = std::make_unique<TopDownCompiler>(pProblem);

  // Set the compiler parameters
  pMDDCompiler->setCompilationType(MDDCompiler::MDDCompilationType::Restricted);
  pMDDCompiler->setNodesRemovalStrategy(
          MDDCompiler::RestrictedNodeSelectionStrategy::CumulativeCost);
  pMDDCompiler->setMaxWidth(width);

  // Start the timer before any computation
  tools::Timer timeoutTimer;

  // Prepare the first layer for the top-down compilation
  buildRootMDD();
  pRootNode->initializeNodeDomain();

  // Find first incumbent
  MDDCompiler::NodePool nodePool;

  // Compile the MDD and get the list of nodes to expand in the next B&B iteration
  pMDDCompiler->compileMDD(pMDDGraph, pArena.get(), nodePool);

  // Get the incumbent, i.e., the best solution found so far
  double bestCost{std::numeric_limits<double>::max()};
  dfsRec(pRootNode, bestCost, static_cast<uint32_t>(pProblem->getVariables().size()),
         0.0, pRootNode);

  if (bestCost < pBestCost)
  {
    pBestCost = bestCost;
    std::cout << "New incumbent: " << pBestCost << " at " <<
            timeoutTimer.getWallClockTimeMsec() << " msec" <<  std::endl;
  }

  // Repeat the compilation process branching on nodes and states and bounding
  // on the bestCost/incumbents found at each iteration
  MDDCompiler::NodePool nodePoolAux;
  bool improveIncumbent{true};
  bool oneEmpty{false};
  while(improveIncumbent)
  {
    if (timeoutTimer.getWallClockTimeMsec() > timeoutMsec)
    {
      // Return on timeout
      break;
    }

    int layerCtr{0};
    for (int currLayer{1}; currLayer < static_cast<int>(nodePool.size()); ++currLayer)
    {
      auto& layerPool = nodePool.at(currLayer);

      // If the current layer is empty, try the next one
      if (layerPool.empty())
      {
        layerCtr++;
        if (layerCtr >= (static_cast<int>(nodePool.size()) - 1))
        {
          // If the pool of nodes is empty, swap with the next one
          // TODO this could loop forever on empty pools
          // improveIncumbent = false;
          nodePool.clear();
          nodePool.swap(nodePoolAux);
          break;
        }
        continue;
      }

      // Found a layer that is not empty, reset the counter
      layerCtr = 0;

      // Get the node from the current layer
      auto topNode = layerPool.top();
      layerPool.pop();

      if (topNode->getDPState()->cumulativeCost() >= pBestCost)
      {
        // Branch the states that are already more expensive
        // than the current incumbent.
        // Note try again with another node on the same layer
        currLayer -= 1;
        continue;
      }

      // Compile the MDD built from the new node:
      // create a new MDD with 1 parent and 1 child until the current node.
      // Use the new MDD on the compiler to find solutions starting from the new node
      rebuildMDDUpToNode(topNode);
      assert(pMDDGraph.at(topNode->getLayer()).at(0) == topNode);

      // Repeat the optimization process on the new MDD
      pMDDCompiler->compileMDD(pMDDGraph, pArena.get(), nodePoolAux);

      // Get the incumbent, i.e., the best solution found so far
      bestCost = std::numeric_limits<double>::max();
      dfsRec(pRootNode, bestCost, static_cast<uint32_t>(pProblem->getVariables().size()),
             0.0, pRootNode);

      if (bestCost < pBestCost)
      {
        pBestCost = bestCost;

        // Set the incumbent on the compiler to avoid storing useless nodes
        pMDDCompiler->setIncumbent(pBestCost);
        std::cout << "New incumbent: " << pBestCost << " at " <<
                timeoutTimer.getWallClockTimeMsec() << " msec" <<  std::endl;
      }
    }
  }
}

void MDDOptimizer::rebuildMDDUpToNode(Node* node)
{
  assert(node != nullptr);

  // Remove all nodes except the root
  for (auto layerIdx{1}; layerIdx < pMDDGraph.size(); ++layerIdx)
  {
    // Remove all nodes from the second layer on (i.e., keep the root)
    for (auto node : pMDDGraph[layerIdx])
    {
      pArena->deleteNode(node->getUniqueId());
    }
    pMDDGraph[layerIdx].clear();
  }

  // Remove root's out edges
  for(auto edge : pRootNode->getOutEdges())
  {
    pArena->deleteEdge(edge->getUniqueId());
  }
  assert(pRootNode->getOutEdges().empty());

  // Rebuild the MDD up to "topNode"
  const auto& path = node->getDPState()->cumulativePath();
  assert(path.size() > 0);

  auto tailNode = pRootNode;
  for (int valIdx{0}; valIdx < static_cast<int>(path.size()); ++valIdx)
  {
    Node* headNode{nullptr};
    if (valIdx < static_cast<int>(path.size()) - 1)
    {
      // Create a new node
      headNode = pArena->buildNode(valIdx+1, (pProblem->getVariables()).at(valIdx+1).get());

      // Set the state
      headNode->resetDPState(tailNode->getDPState()->next(path.at(valIdx)));
    }
    else
    {
      // Use "topNode" as last node
      headNode = node;
    }
    pArena->buildEdge(tailNode, headNode, path.at(valIdx), path.at(valIdx));
    pMDDGraph[valIdx+1].push_back(headNode);

    // Swap nodes
    tailNode = headNode;
  }
}

void MDDOptimizer::buildRootMDD()
{
  // Build the root node
  pMDDGraph.at(kLayerZero).clear();
  pMDDGraph.at(kLayerZero).push_back(
          pArena->buildNode(kLayerZero, pProblem->getVariables().at(kLayerZero).get()));
  pRootNode = pMDDGraph.at(kLayerZero).back();
}

void MDDOptimizer::dfsRec(Node* currNode, double& bestCost, const uint32_t maxLayer,
                          double cost, Node* prevNode, bool debug)
{
  const auto& edges = currNode->getOutEdges();
  if (edges.empty())
  {
    if (currNode->getLayer() < maxLayer)
    {
      return;
    }

    if (cost < bestCost)
    {
      bestCost = cost;
    }
    return;
  }

  for (auto it = edges.begin(); it != edges.end(); ++it)
  {
    if (debug)
    {
      std::cout << "-> " << (*it)->getValue() << "(" <<
              currNode->getDPState()->cost((*it)->getValue(), prevNode->getDPState()) << ") ";
    }
    cost += currNode->getDPState()->cost((*it)->getValue(), prevNode->getDPState());
    dfsRec((*it)->getHead(), bestCost, maxLayer, cost, currNode, debug);
  }
}

void MDDOptimizer::printMDD(const std::string& outFileName)
{
  std::string ppMDD = "digraph D {\n";
  for (int lidx{0}; lidx < static_cast<int>(pMDDGraph.size()); ++lidx)
  {
    for (auto node : pMDDGraph[lidx])
    {
      for (auto outEdge : node->getOutEdges())
      {
        std::string toNodeId = outEdge->getHead()->getNodeStringId();
        if (lidx == static_cast<int>(pMDDGraph.size()) - 2)
        {
          toNodeId = "t";
        }
        std::string newEdge = node->getNodeStringId() + " -> " + toNodeId;
        newEdge += std::string("[label=") + "\"" + std::to_string(outEdge->getValue())  +  "\"]\n";
        ppMDD += "\t" + newEdge;
      }
    }
  }
  ppMDD += "}";

 std::ofstream outFile;
 outFile.open(outFileName + ".dot");
 outFile << ppMDD;
 outFile.close();
}

}  // namespace mdd
