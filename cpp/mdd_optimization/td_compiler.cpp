#include "mdd_optimization/td_compiler.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument

#include <sparsepp/spp.h>

namespace {
bool cmpStateList(const std::unique_ptr<mdd::DPState>& a, const std::unique_ptr<mdd::DPState>& b)
{
  return a->cumulativeCost() < b->cumulativeCost();
}
}  // namespace

namespace mdd {

TDCompiler::TDCompiler(MDDProblem::SPtr problem, uint32_t width)
: pProblem(problem)
{
  if (problem == nullptr)
  {
    throw std::invalid_argument("TDCompiler: empty pointer to the problem instance");
  }

  // Initialize the MDD.
  // TODO evaluate performance of storing two MDDs:
  // 1) MDD for relaxed compilation;
  // 2) MDD for restricted compilation
  pMDDGraph = std::make_unique<TopDownMDD>(pProblem, width);
}

bool TDCompiler::compileMDD(CompilationMode compilationMode)
{
  // Start from the root node and build the MDD Top-Down.
  // Building the MDD means replacing the states on the nodes AND
  // activating the edge connections between nodes.
  // Note: the root node state has been already set to the default state
  // when building the MDD
  return buildMDD(compilationMode);
}

bool TDCompiler::buildMDD(CompilationMode compilationMode)
{
  const auto numLayers = pMDDGraph->getNumLayers();
  for (uint32_t lidx{0}; lidx < numLayers; ++lidx)
  {
    // Use a flag to skip layers that were previously built.
    // This can happen, for example, when running Top-Down on an MDD
    // that has been built from a state rather than from the root,
    // e.g., during branch and
    bool layerHasSuccessors{false};

    // Get the variable associated with the current layer.
    // Note: the variable is on the tail of the edge on the current layer
    auto var = pMDDGraph->getVariablePerLayer(lidx);

    // Go over each node on the current layer and calculate next layer's nodes
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
          // Break and proceed with next layer
          layerHasSuccessors = true;
          break;
        }
      }

      // Compute all next states from current node and filter only the "best" states.
      // Note: the original restricted algorithm, given layer i, first creates all 'k' states
      // for layer i+1; then it shrinks them to 'w' according to a selection heuristic.
      // In what follows, we create only the states that will be part of layer i+1.
      // In other words, instead of computing all 'k' states, the algorithm selects the 'w'
      // best states at each iteration and overrides them
      std::vector<std::pair<MDDTDEdge*, bool>> newActiveEdgeList;
      if (compilationMode == CompilationMode::Relaxed)
      {
        // Compute all next states from current node and merge the states.
        // Note: the original relaxed algorithm, given layer i, first creates all 'k' states
        // for layer i+1; then it shrinks them to 'w' according to a merge heuristic.
        // In what follows, we create only the states that will be part of layer i+1.
        // In other words, instead of computing all 'k' states, the algorithm
        // compute 'k'costs used by the heuristic, selects the 'w' best costs, and creates
        // only those
        newActiveEdgeList = relaxNextLayerStatesFromNode(lidx, nidx,
                                                         var->getLowerBound(),
                                                         var->getUpperBound());

      }
      else if (compilationMode == CompilationMode::Restricted)
      {
        // Compute all next states from current node and filter only the "best" states.
        // Note: the original restricted algorithm, given layer i, first creates all 'k' states
        // for layer i+1; then it shrinks them to 'w' according to a selection heuristic.
        // In what follows, we create only the states that will be part of layer i+1.
        // In other words, instead of computing all 'k' states, the algorithm selects the 'w'
        // best states at each iteration and overrides them
        newActiveEdgeList = restrictNextLayerStatesFromNode(lidx, nidx,
                                                            var->getLowerBound(),
                                                            var->getUpperBound());
      }

      // All edges have "nidx" as tail node and lead to a node on layer "lidx+1"
      for (const auto& edgePair : newActiveEdgeList)
      {
        assert(edgePair.first->tail == nidx);

        // This layer has at least one successor on the next layer
        layerHasSuccessors = true;
        if (edgePair.second)
        {
          // Deactivate all current layers and active given layer
          for (auto edge : pMDDGraph->getEdgeOnHeadMutable(lidx, edgePair.first->head))
          {
            edge->isActive = false;
          }
        }

        // Activate given layer
        edgePair.first->isActive = true;
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

std::vector<std::pair<MDDTDEdge*, bool>> TDCompiler::restrictNextLayerStatesFromNode(
        uint32_t currLayer, uint32_t currNode, int64_t lb, int64_t ub)
{
  std::vector<std::pair<MDDTDEdge*, bool>> newConnections;

  // Get the start node
  auto currState = pMDDGraph->getNodeState(currLayer, currNode);
  assert(currState != nullptr);

  // Get the list of next states to override (since only "width" states are allowed)
  const auto nextLayer = currLayer + 1;
  TopDownMDD::StateList* nextStateList = pMDDGraph->getStateListMutable(nextLayer);

  // Get the list of best "width" next states reachable from the current state
  // according to the heuristic implemented in the DP model
  auto stateList = currState->nextStateList(lb, ub, getIncumbent());

  // Check whether or not to use next states
  const auto width = static_cast<uint32_t>(nextStateList->size());
  std::sort(stateList.begin(), stateList.end(), cmpStateList);

  // First replace all default states
  uint32_t repPtr{0};
  auto defaultStateIdx = pMDDGraph->getIndexOfFirstDefaultStateOnLayer(nextLayer);
  while (defaultStateIdx < width)
  {
    if (repPtr >= static_cast<uint32_t>(stateList.size()))
    {
      // No more replaceable states, break
      break;
    }
    assert(stateList.at(repPtr)->cumulativeCost() <= getIncumbent());

    // Replace the state in the MDD
    const auto val = stateList.at(repPtr)->cumulativePath().back();
    pMDDGraph->replaceState(nextLayer, defaultStateIdx, std::move(stateList[repPtr]));

    // Activate a new edge
    auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, defaultStateIdx);
    edge->valuesList[0] = val;
    newConnections.emplace_back(edge, true);

    // Move to next state
    ++repPtr;

    // Update index to next default state
    defaultStateIdx = pMDDGraph->getIndexOfFirstDefaultStateOnLayer(nextLayer);
  }

  // If there are still nodes left, then override non-default states
  while(repPtr < static_cast<uint32_t>(stateList.size()))
  {
    // Here all default states are replaced.
    // Therefore, start replacing nodes from the beginning (wrapping around)
    for (uint32_t idx{0}; idx < width; ++idx)
    {
      assert(!nextStateList->at(idx)->isDefaultState());

      // Check whether or not merge same state
      auto currState = stateList.at(repPtr).get();
      if (nextStateList->at(idx)->isEqual(currState))
      {
        // If two states are equal, they can be merged,
        // i.e., the correspondent edge can be activated.
        // Note: two nodes can be equal but have a different cost.
        // The cost is set to the lower of the two, following
        // the heuristic of keeping low costs nodes only at each layer
        if (nextStateList->at(idx)->cumulativeCost() > currState->cumulativeCost())
        {
          nextStateList->at(idx)->forceCumulativeCost(currState->cumulativeCost());
        }

        // Merge a new edge
        auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, idx);
        assert(edge != nullptr);

        if (!edge->hasValueSet())
        {
          edge->valuesList[0] = currState->cumulativePath().back();
        }
        else
        {
          edge->valuesList.push_back(currState->cumulativePath().back());
        }
        newConnections.emplace_back(edge, false);

        break;
      }

      // Check if new cost is lower than the current one.
      // If so, replace the state.
      // This is equivalent to the original restricted algorithm
      // where states are first all created and then pruned to keep
      // the width contained.
      // Here, instead of pruning, states are overridden
      if (nextStateList->at(idx)->cumulativeCost() > currState->cumulativeCost())
      {
        // The state is overridden by the new one
        // Replace the state in the MDD
        const auto val = stateList[repPtr]->cumulativePath().back();
        pMDDGraph->replaceState(nextLayer, idx, std::move(stateList[repPtr]));

        // Activate a new edge
        auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, idx);
        edge->valuesList[0] = val;
        newConnections.emplace_back(edge, true);

        break;
      }
    }

    // Check next state
    ++repPtr;
  }

  return newConnections;
}

std::vector<std::pair<MDDTDEdge*, bool>> TDCompiler::relaxNextLayerStatesFromNode(
        uint32_t currLayer, uint32_t currNode, int64_t lb, int64_t ub)
{
  std::vector<std::pair<MDDTDEdge*, bool>> newConnections;

  // Get the start node
  auto currState = pMDDGraph->getNodeState(currLayer, currNode);
  assert(currState != nullptr);

  // Get the list of next states to override (since only "width" states are allowed)
  const auto nextLayer = currLayer + 1;
  TopDownMDD::StateList* nextStateList = pMDDGraph->getStateListMutable(nextLayer);

  // Get the list of best "width" next states reachable from the current state
  // according to the heuristic implemented in the DP model
  auto stateList = currState->nextStateList(lb, ub,  getIncumbent());

  // Check whether or not to use next states
  const auto width = static_cast<uint32_t>(nextStateList->size());
  std::sort(stateList.begin(), stateList.end(), cmpStateList);

  // First replace all default states
  uint32_t repPtr{0};
  auto defaultStateIdx = pMDDGraph->getIndexOfFirstDefaultStateOnLayer(nextLayer);
  while (defaultStateIdx < width)
  {
    if (repPtr >= static_cast<uint32_t>(stateList.size()))
    {
      // No more replaceable states, break
      break;
    }
    assert(stateList.at(repPtr)->cumulativeCost() <= getIncumbent());

    // Replace the state in the MDD
    const auto val = stateList.at(repPtr)->cumulativePath().back();
    pMDDGraph->replaceState(nextLayer, defaultStateIdx, std::move(stateList[repPtr]));

    // Activate a new edge
    auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, defaultStateIdx);
    edge->valuesList[0] = val;
    newConnections.emplace_back(edge, true);

    // Move to next state
    ++repPtr;

    // Update index to next default state
    defaultStateIdx = pMDDGraph->getIndexOfFirstDefaultStateOnLayer(nextLayer);
  }

  // If there are still nodes left, then override non-default states
  while(repPtr < static_cast<uint32_t>(stateList.size()))
  {
    auto currState = stateList.at(repPtr).get();

    // Check first if one of the states already present is equivalent
    // to the current state
    bool foundEquivalent{false};
    for (uint32_t idx{0}; idx < width; ++idx)
    {
      assert(!nextStateList->at(idx)->isDefaultState());
      if (nextStateList->at(idx)->isEqual(currState))
      {
        // If two states are equal, they can be merged,
        // i.e., the correspondent edge can be activated.
        // Note: two nodes can be equal but have a different cost.
        // The cost is set to the lower of the two, following
        // the heuristic of keeping low costs nodes only at each layer
        if (nextStateList->at(idx)->cumulativeCost() > currState->cumulativeCost())
        {
          nextStateList->at(idx)->forceCumulativeCost(currState->cumulativeCost());
        }

        // Merge a new edge
        auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, idx);
        assert(edge != nullptr);

        if (!edge->hasValueSet())
        {
          edge->valuesList[0] = currState->cumulativePath().back();
        }
        else
        {
          edge->valuesList.push_back(currState->cumulativePath().back());
        }

        newConnections.emplace_back(edge, false);
        foundEquivalent = true;

        break;
      }
    }

    if (foundEquivalent)
    {
      // Found an equivalent state, continue checking next state
      ++repPtr;
      continue;
    }

    // Pick one of the state to merge the current one into
    auto stateIdx = currState->stateSelectForMerge(*nextStateList);

    // Merge the state
    const auto val = currState->cumulativePath().back();
    (*nextStateList)[stateIdx]->mergeState(currState);

    auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, stateIdx);
    if (!edge->hasValueSet())
    {
      edge->valuesList[0] = val;
    }
    else
    {
      edge->valuesList.push_back(val);
    }
    newConnections.emplace_back(edge, false);

    // Continue with next state
    ++repPtr;
  }

  return newConnections;
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
