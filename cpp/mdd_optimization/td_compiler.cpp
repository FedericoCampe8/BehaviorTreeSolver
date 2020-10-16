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

bool TDCompiler::compileMDD(CompilationMode compilationMode, DPState::UPtr state)
{
  // Reset exact MDD flag
  pIsExactMDD = true;

  // Reset the graph
  if (state != nullptr)
  {
    // Reset the MDD
    pMDDGraph->resetGraph();

    // Build the MDD up to "state"
    buildMDDUpToState(std::move(state));
  }

  // Start from the root node and build the MDD Top-Down.
  // Building the MDD means replacing the states on the nodes AND
  // activating the edge connections between nodes.
  // Note: the root node state has been already set to the default state
  // when building the MDD
  return buildMDD(compilationMode);
}

void TDCompiler::buildMDDUpToState(DPState::UPtr node)
{
  // Start from the root node
  auto tailNode = pMDDGraph->getNodeState(0, 0);

  // Follow the path to re-build
  const auto& path = node->cumulativePath();
  for (int valIdx{0}; valIdx < static_cast<int>(path.size()); ++valIdx)
  {
    // Create a new node
    auto headNode = pMDDGraph->getNodeState(valIdx+1, 0);
    headNode->updateState(tailNode, path.at(valIdx));
    headNode->setNonDefaultState();

    // Activate the corresponding edge
    auto edge = pMDDGraph->getEdgeMutable(valIdx, 0, 0);
    edge->isActive = true;
    edge->valuesList[0] = path.at(valIdx);

    // Swap nodes
    tailNode = headNode;
  }
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
    uint32_t numGenStates{0};
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
                                                            var->getUpperBound(),
                                                            numGenStates);
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
    }  // for all nodes

    // Check for exact MDDs
    if (compilationMode == CompilationMode::Restricted)
    {
      // The MDD is NOT exact if a layer exceeds the width
      if (numGenStates > pMDDGraph->getMaxWidth())
      {
        pIsExactMDD = false;
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
        uint32_t currLayer, uint32_t currNode, int64_t lb, int64_t ub, uint32_t& generatedStates)
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
  generatedStates += static_cast<uint32_t>(stateList.size());

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
    pMDDGraph->getNodeState(nextLayer, defaultStateIdx)->setNonDefaultState();

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
  //std::cout << "TOP DOWN ON " << currLayer << " " << currNode << std::endl;
  //std::cout << currState->toString() << std::endl; getchar();
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
    const bool isExact = stateList[repPtr]->isExact();
    pMDDGraph->replaceState(nextLayer, defaultStateIdx, std::move(stateList[repPtr]));
    pMDDGraph->getNodeState(nextLayer, defaultStateIdx)->setExact(isExact);

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
        // Check if the two states are strictly equal to set
        // the exact state
        if (nextStateList->at(idx)->isStrictlyEqual(currState))
        {
          nextStateList->at(idx)->setExact(nextStateList->at(idx)->isExact() &&
                                           currState->isExact());
        }

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
    (*nextStateList)[stateIdx]->setExact(false);
    (*nextStateList)[stateIdx]->setNonDefaultState();

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

}  // namespace mdd
