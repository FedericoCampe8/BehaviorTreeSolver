#include "mdd_optimization/td_compiler.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument

#include <sparsepp/spp.h>

namespace {
constexpr uint32_t kLayerZero{0};
}  // namespace

namespace mdd {

TDCompiler::TDCompiler(MDDProblem::SPtr problem, uint32_t width)
: pProblem(problem)
{
  if (problem == nullptr)
  {
    throw std::invalid_argument("TDCompiler: empty pointer to the problem instance");
  }

  // Initialize the MDD
  pMDDGraph = std::make_unique<TopDownMDD>(pProblem, width);
}

bool TDCompiler::compileMDD()
{
  // Start from the root node and build the MDD Top-Down.
  // Building the MDD means replacing the states on the nodes AND
  // activating the edge connections between nodes.
  // Note: the root node state has been already set to the default state
  // when building the MDD
  const auto numLayers = pMDDGraph->getNumLayers();
  for (uint32_t lidx{0}; lidx < numLayers; ++lidx)
  {
    bool layerHasSuccessors{false};
    auto var = pMDDGraph->getVariablePerLayer(lidx);
    for (uint32_t nidx{0}; nidx < pMDDGraph->getMaxWidth(); ++nidx)
    {
      if (lidx == 0 && nidx > 0)
      {
        // Break on layer 0 after the first node since layer 0
        // contains only the root
        break;
      }

      // Given the current state, for each domain element compute next state
      // and add it to the MDD.
      // Adding a state means:
      // a) replacing existing state;
      // b) activating the edge connecting current to next state;
      // c) setting the value on the edge
      if (!pMDDGraph->isReachable(lidx, nidx))
      {
        // Skip non reachable states
        continue;
      }

      // Check if this MDD has a non-default state as first state
      // in the next layer when considering the first state on the current layer.
      // If so, it means that the MDD was "re-built" and next layer shouldn't be touched
      if (nidx == 0)
      {
        const auto defaultStateIdx = pMDDGraph->getIndexOfFirstDefaultStateOnLayer(lidx + 1);
        if (defaultStateIdx > 0)
        {
          // Break, proceed with next layer
          layerHasSuccessors = true;
          break;
        }
      }

      // Compute all next states from current node and filter only
      // the "best" states.
      // "Best" states are selected with the heuristic implemented
      // in the DP constraint
      auto replacedStates = calculateNextLayerStates(lidx, nidx,
                                                     var->getLowerBound(),
                                                     var->getUpperBound());

      // For each replaced state:
      // I)   disable current arc;
      // II)  enable arc with tail equal to "nidx"
      // III) set the value on the arc to be equal to the replaced value
      for (const auto& repState : replacedStates)
      {
        // This layer has at least one successor on the next layer
        layerHasSuccessors = true;

        // Enable the edges to connect the current layer to the next one
        const auto idxHeadNodeToReplace = repState.first;
        const auto edgeValue = repState.second;

        // Note: the edge can be nullptr if there is no current active edge
        // incoming to the state
        auto currEdge = pMDDGraph->getEdgeOnHeadMutable(lidx, idxHeadNodeToReplace);
        if (currEdge != nullptr)
        {
          // Disable the current active edge (if any)
          const auto currTailEdge = currEdge->tail;
          pMDDGraph->disableEdge(lidx, currTailEdge, idxHeadNodeToReplace);
        }

        // Enable the edge on layer "lidx" from node "nidx" to node " idxNodeReplaced"
        pMDDGraph->enableEdge(lidx, nidx, idxHeadNodeToReplace);

        // Set value on the newly enabled edge
        pMDDGraph->setEdgeValue(lidx, nidx, idxHeadNodeToReplace, edgeValue);
      }
    }

    if (!layerHasSuccessors)
    {
      // No successor to next layer, break since there are no solution
      return false;
    }
  }
  return true;
}

DPState::ReplacementNodeList TDCompiler::calculateNextLayerStates(
        uint32_t currLayer, uint32_t currNode, int64_t lb, int64_t ub)
{
  auto currState = pMDDGraph->getNodeState(currLayer, currNode);
  assert(currState != nullptr);

  const auto nextLayerIdx = currLayer + 1;

  // Get the list of next states
  TopDownMDD::StateList* nextStateList = pMDDGraph->getStateListMutable(nextLayerIdx);

  // Get all the values reachable from the current state
  auto costList = currState->getCostListPerValue(lb, ub, getIncumbent());

  // Keep track of the replaced states and values
  DPState::ReplacementNodeList replacedStates;
  if (costList.empty())
  {
    // No values can change the current states.
    // Return asap
    return replacedStates;
  }

  // "costList" contains the list of all values that can lead to admissible new states.
  // Sort the list to return the best states
  std::sort(costList.begin(), costList.end());

  // Heuristic: replace in a new state ONLY IF the new state has a cost lower than
  // the current states in the layer.
  // Therefore, calculate the minimum value on the current list of states
  const uint32_t width = static_cast<uint32_t>(nextStateList->size());

  // States are sorted: start from the lowest cost state
  uint32_t repPtr{0};

  // First replace all default states (before overriding non-default states)
  const auto costListSize = static_cast<uint32_t>(costList.size());
  auto defaultStateIdx = pMDDGraph->getIndexOfFirstDefaultStateOnLayer(nextLayerIdx);
  while (defaultStateIdx < width)
  {
    if (repPtr >= costListSize)
    {
      // No more replaceable states, break
      break;
    }

    if (costList.at(repPtr).first >= getIncumbent())
    {
      // Exclude costs greater than the current incumbent
      break;
    }

    // Replace the current state and keep track of:
    // 1) the value to assign to the edge; and
    // 2) what state was replaced (to disable the corresponding incoming edge)
    // Note: in general they might be multiple incoming edges however:
    //  a) if the state is default, no incoming edge is active; or
    //  b) if the state is not default, only one incoming edge is active
    //     (the one leading to the state)
    const auto newEdgeValue = costList.at(repPtr).second;
    replacedStates.push_back({defaultStateIdx, newEdgeValue});

    // Replace the state in the MDD
    pMDDGraph->replaceState(nextLayerIdx, defaultStateIdx, currState, newEdgeValue, false,
                            getIncumbent());

    // Move to next state
    ++repPtr;

    // Update index to next default state
    defaultStateIdx = pMDDGraph->getIndexOfFirstDefaultStateOnLayer(nextLayerIdx);
  }

  // Then override non-default states if there are any new states in "costList"
  // left to consider
  if (repPtr < costListSize)
  {
    for (uint32_t idx{0}; idx < width; ++idx)
    {
      if (repPtr >= costListSize)
      {
        // No more replaceable states, break
        break;
      }

      if (costList.at(repPtr).first >= getIncumbent())
      {
        // Exclude costs greater than the current incumbent
        break;
      }

      // Try to replace the current state with one having a lower cost
      if (nextStateList->at(idx)->cumulativeCost() > costList.at(repPtr).first)
      {
        // Replace the current state and keep track of:
        // 1) the value to assign to the edge; and
        // 2) what state was replaced (to disable the corresponding incoming edge)
        // Note: in general they might be multiple incoming edges however:
        //  a) if the state is default, no incoming edge is active; or
        //  b) if the state is not default, only one incoming edge is active
        //     (the one leading to the state)
        const auto newEdgeValue = costList.at(repPtr).second;
        replacedStates.push_back({idx, newEdgeValue});

        // Replace the state in the MDD
        pMDDGraph->replaceState(nextLayerIdx, idx, currState, newEdgeValue, true, getIncumbent());

        // Move to next state
        ++repPtr;
      }
    }
  }

  // Check if there are some possible states left that didn't replace
  // the current states
  if (repPtr < costListSize)
  {
    // If so, store these states in the queue to pick them up later on
    // when doing branch & bound (otherwise these state will be discarded forever)
    while (repPtr < costListSize)
    {
      if (costList.at(repPtr).first >= getIncumbent())
      {
        ++repPtr;
        continue;
      }
      pMDDGraph->buildAndStoreState(nextLayerIdx, currState, costList.at(repPtr).second);

      // Get next state and repeat
      ++repPtr;
    }
  }

  // Return the replacement list
  return replacedStates;
}

bool TDCompiler::rebuildMDDFromQueue()
{
  // Store all the nodes in the MDD that are leaf nodes,
  // i.e., not part of the path to the tail node but still on the MDD
  // and not replaced, i.e., not in the queue of stored stated
  // TODO double-check if this is required
  // pMDDGraph->storeLeafNodes(getIncumbent());

  // Check if the MDD can be recompiled.
  // The MDD can be recompiled if there is at least
  // one state stored in the queue of states
  if (!pMDDGraph->hasStoredStates())
  {
    return false;
  }

  // Reset all MDD
  const bool resetStatesQueue{false};
  pMDDGraph->resetGraph(resetStatesQueue);

  // Build the MDD from the queued states
  pMDDGraph->rebuildMDDFromStoredStates(getIncumbent());

  // Return success
  return true;
}


#ifdef OLDCODE
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
#endif

}  // namespace mdd
