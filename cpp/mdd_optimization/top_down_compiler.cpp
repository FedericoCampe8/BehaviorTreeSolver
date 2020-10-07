#include "mdd_optimization/top_down_compiler.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <cstdlib>    // for std::rand
#include <iostream>
#include <stdexcept>  // for std::invalid_argument

#include <sparsepp/spp.h>

namespace {
constexpr uint32_t kLayerZero{0};
}  // namespace

namespace mdd {

TopDownCompiler::TopDownCompiler(MDDProblem::SPtr problem)
: MDDCompiler(MDDCompiler::MDDConstructionAlgorithm::TopDown),
  pProblem(problem)
{
  if (problem == nullptr)
  {
    throw std::invalid_argument("TopDownCompiler - compileMDD: nullptr to problem");
  }

  // Initialize random seed
  srand((unsigned)time(NULL));
}

void TopDownCompiler::compileMDD(MDDCompiler::MDDGraph& mddGraph, Arena* arena, NodePool& nodePool)
{
  if (arena == nullptr)
  {
    throw std::invalid_argument("TopDownCompiler - compileMDD: nullptr to arena");
  }

  if (mddGraph.empty() || mddGraph.at(0).empty())
  {
    throw std::invalid_argument("TopDownCompiler - compileMDD: MDD root not found");
  }

  // Initialize internal members
  pArena = arena;

  // Build the MDD
  buildTopDownMDD(mddGraph, nodePool);
}

void TopDownCompiler::buildTopDownMDD(MDDGraph& mddGraph, NodePool& nodePool)
{
  // The Top-Down procedure assumes that there is ONLY ONE DP model
  // encoded as a constraint that represents the problem
  if (getConstraintsList().size() != 1)
  {
    throw std::runtime_error("TopDownCompiler - buildTopDownMDD: the model should be defined by "
            "one and only one Dynamic Programming model");
  }
  auto dpConstraint = getConstraintsList().at(0).get();

  // Set the initial DP state on the root node to activate the DP chain
  mddGraph.at(kLayerZero).at(0)->resetDPState(dpConstraint->getInitialDPState());

  // Get the total number of layers of the MDD which is the same as the number of variable.
  // Note: the number of layers must be taken from the number of variable and NOT from the
  // given graph. Indeed, the given graph should contain only the initial root state
  const auto totLayers = static_cast<int>(pProblem->getVariables().size());

  // Create the ordered queue to keep track of the discarded nodes on each layer
  if (nodePool.size() < totLayers)
  {
    nodePool.resize(totLayers);
  }

  // Build the list of new states for comparing states to merge when running
  // exact compilation
  std::vector<Node*> newDPStates;
  newDPStates.reserve(std::min(32, getMaxWidth()));

  // Proceed one layer at a time top-down
  for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
  {
    // Reset the list of new states
    newDPStates.clear();

    // Merge nodes if compilation type is relaxed
    if (getCompilationType() == MDDCompiler::MDDCompilationType::Relaxed)
    {
      // Apply merging procedure for relaxed MDDs
      mergeNodes(layerIdx, mddGraph);
    }

    if (getCompilationType() == MDDCompiler::MDDCompilationType::Restricted)
    {
      // Apply restricted procedure to remove nodes from the MDD
      if (mddGraph.at(layerIdx).size() > getMaxWidth())
      {
        removeNodes(layerIdx, mddGraph, nodePool);
      }
    }

    // For all nodes per layer
    for (int nodeIdx{0}; nodeIdx < static_cast<int>(mddGraph.at(layerIdx).size()); ++nodeIdx)
    {
      // For all values of the domain of the current layer
      auto currNode = mddGraph.at(layerIdx).at(nodeIdx);

      // Handle the corner case where the MDD is already build:
      // On the first node of the current layer, check the following layer.
      // If it has been built already, skip it
      const auto nextLayer = currNode->getLayer() + 1;
      const auto nextLayerSize = mddGraph.at(nextLayer).size();
      if (nodeIdx == 0 && nextLayerSize > 0)
      {
        // Next layer already has some nodes, continue.
        // Note: next layer should have AT MOST one node
        if (nextLayerSize > 1)
        {
          std::cerr << "TopDownCompiler - buildTopDownMDD: ERROR - layer " << nextLayer <<
                  " has more than 1 node before compilation. It has " << nextLayerSize <<
                  " nodes." << std::endl;
        }
        continue;
      }

      // Expand on the next node BUT do not build a new node per domain element,
      // but build only "width" admissible good nodes
      const auto* var = currNode->getVariable();
      auto nextLayerStates = currNode->getDPState()->next(var->getLowerBound(),
                                                          var->getUpperBound(),
                                                          getMaxWidth(),
                                                          getBestCost());
      if (nextLayerStates.empty() || nextLayerStates.at(0) == nullptr)
      {
        // This node cannot lead to anything
        continue;
      }

      // Create a new now for each state and connect it to the current node,
      // except the last state which goes into the queue
      for (int sidx{0}; sidx < static_cast<int>(nextLayerStates.size()) - 1; ++sidx)
      {
        // Create a new node for the current state.
        // The new node should own the variable on next layer
        const auto nextLayer = currNode->getLayer() + 1;
        auto nextVar = nextLayer < (getVariablesList()).size() ?
                getVariablesList().at(nextLayer).get() :
                nullptr;
        auto nextNode = pArena->buildNode(currNode->getLayer() + 1, nextVar);

        // Set the new DP state on the new node
        nextNode->resetDPState(nextLayerStates.at(sidx));

        // Create an edge connecting the current node and the next node
        const auto val = nextNode->getDPState()->cumulativePath().back();
        pArena->buildEdge(currNode, nextNode, val, val);

        // Add the node to the next layer
        mddGraph.at(nextLayer).push_back(nextNode);
      }

      // Create last node to put in the queue
      auto& lastState = nextLayerStates.back();
      if (lastState != nullptr)
      {
        // This node represents all the choices that are filtered (but still valid)
        // on calling "next" above.
        // Note: the new node will replace "currNode" once the MDD is re-built.
        //       Therefore it must point to its variable and be on the current layer
        auto filterNode = pArena->buildNode(currNode->getLayer(), currNode->getVariable());
        filterNode->resetDPState(lastState);
        nodePool.at(filterNode->getLayer()).push(filterNode);
      }
    }  // for all nodes in a given layer
  }  // for all layers in the MDD
}

void TopDownCompiler::mergeNodes(int layer, MDDGraph& mddGraph)
{
  if (layer == 0)
  {
    // The first layer has the root node and doesn't need to be merged
    return;
  }

  while(mddGraph.at(layer).size() > getMaxWidth())
  {
    // Step I: select a subset of nodes to merge
    auto subsetNodes = (getConstraintsList().at(0))->mergeNodeSelect(layer, mddGraph);
    assert(subsetNodes.size() > 1);

    // Step II: remove the selected nodes from the MDD
    std::vector<Edge*> edgesToRedirect;
    spp::sparse_hash_set<uint32_t> mergingNodes;
    auto& currentLevel = mddGraph[layer];

    for (auto node : subsetNodes)
    {
      // Check that there are no duplicate nodes
      if (mergingNodes.find(node->getUniqueId()) != mergingNodes.end())
      {
        throw std::runtime_error("TopDownCompiler - mergeNodes: "
                "duplicate nodes selected for merging");
      }
      mergingNodes.insert(node->getUniqueId());

      auto it = std::find(currentLevel.begin(), currentLevel.end(), node);
      assert(it != currentLevel.end());

      // Keep track of the incoming edges to merge
      for (auto inEdge : (*it)->getInEdges())
      {
        edgesToRedirect.push_back(inEdge);
      }

      // Remove the node from the level
      currentLevel.erase(it);

      // Step III: Merge all the nodes into one and get their representative
      auto newMergedNode = (getConstraintsList().at(0))->mergeNodes(subsetNodes, pArena);

      // Step IV: Re-route all the edges to the new node
      for (auto redirectEdge : edgesToRedirect)
      {
        redirectEdge->setHead(newMergedNode);
      }

      // Step V: Delete the merged nodes
      for (auto node : subsetNodes)
      {
        assert(node->getOutEdges().empty());
        pArena->deleteNode(node->getUniqueId());
      }
    }
  }
}

void TopDownCompiler::removeNodes(int layer, MDDGraph& mddGraph, NodePool& nodePool)
{
  if (getMaxWidth() < 2)
  {
    // Cannot remove nodes if the max-width is set to be less than two
    return;
  }

  while(mddGraph.at(layer).size() > getMaxWidth())
  {
    // Select the nodes to remove
    int nodeToRemoveIdx{0};
    switch (getNodesRemovalStrategy())
    {
      case MDDCompiler::RestrictedNodeSelectionStrategy::CumulativeCost:
      {
        double worstCost{0};
        for (int nidx{0}; nidx < static_cast<int>(mddGraph.at(layer).size()); ++nidx)
        {
          auto currNode = mddGraph.at(layer).at(nidx);
          assert((currNode->getInEdges()).size() == 1);

          const auto currCost = currNode->getDPState()->cumulativeCost();
          if (currCost > worstCost)
          {
            worstCost = currCost;
            nodeToRemoveIdx = nidx;
          }
        }
        break;
      }
      case MDDCompiler::RestrictedNodeSelectionStrategy::RightToLeft:
      {
        nodeToRemoveIdx = static_cast<int>(mddGraph.at(layer).size()) - 1;
        break;
      }
      case MDDCompiler::RestrictedNodeSelectionStrategy::LeftToRight:
      {
        nodeToRemoveIdx = 0;
        break;
      }
      case MDDCompiler::RestrictedNodeSelectionStrategy::Random:
      {
        nodeToRemoveIdx = std::rand() % static_cast<int>((mddGraph.at(layer).size()));
        break;
      }
    }

    // Get the node and remove it from the MDD
    auto lastNode = mddGraph.at(layer).at(nodeToRemoveIdx);
    mddGraph.at(layer).erase(mddGraph.at(layer).begin() + nodeToRemoveIdx);

    // Remove all the connected edges as well
    assert(lastNode->getOutEdges().empty());
    auto allEdges = lastNode->getInEdges();
    for (auto inEdgeToRemove : allEdges)
    {
      pArena->deleteEdge(inEdgeToRemove->getUniqueId());
    }

    // Store the node to be used later on during branch and bound
    // Note: do not store nodes which value is already greater than the best incumbent
    if (lastNode->getDPState()->cumulativeCost() < getBestCost())
    {
      nodePool.at(lastNode->getLayer()).push(lastNode);
    }
    else
    {
      // Delete the node if not stored
      pArena->deleteNode(lastNode->getUniqueId());
    }
  }
}

}  // namespace mdd
