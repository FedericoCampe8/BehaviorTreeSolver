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
  return buildMDD2(compilationMode);
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
    edge->costList[0] = tailNode->getCostOnValue(path.at(valIdx));

    // Swap nodes
    tailNode = headNode;
  }
}

bool TDCompiler::buildMDD2(CompilationMode compilationMode)
{
  // Process one layer at a time
  bool setEdgesForCutset{false};
  for (uint32_t lidx{0}; lidx < pMDDGraph->getNumLayers(); ++lidx)
  {
    // Skip layers that were previously built when reconstructing the MDD
    // from a node that was in the queue.
    // This layers have a non-default state as the first state in that layer.
    // Every other MDD (e.g., the first one being built) contains only default states
    if (pMDDGraph->getIndexOfFirstDefaultStateOnLayer(lidx + 1))
    {
      continue;
    }

    bool isValidMDD{false};
    if (compilationMode == CompilationMode::Restricted)
    {
      isValidMDD = buildRestrictedMDD(lidx);
    }
    else if (compilationMode == CompilationMode::Relaxed)
    {
      isValidMDD = buildRelaxedMDD(lidx, setEdgesForCutset);
    }

    if (!isValidMDD)
    {
      return false;
    }
  }
  return true;
}

bool TDCompiler::buildRestrictedMDD(uint32_t layer)
{
  // Get the variable associated with the current layer.
  // @note the variable is on the tail of the edge on the current layer
  auto var = pMDDGraph->getVariablePerLayer(layer);

  // Go over each node on the current layer and calculate next layer's nodes.
  // @note keep track of:
  // 1 - whether or not the restricted MDD is exact;
  // 2 - if a layer is connected to the next one.
  // An MDD is exact (1) if, when compiled as restricted MDD, the list of "next" nodes
  // from a layer never exceeds the width of the MDD.
  // If a layer is not connected to the next one (2), the MDD doesn't contain any path root-tail.
  // An MDD may not be connected if, given a layer, all next nodes lead to nodes
  // with a higher cost than the current incumbent
  uint32_t numGenStates{0};
  bool layerHasSuccessors{false};
  for (uint32_t nidx{0}; nidx < pMDDGraph->getMaxWidth(); ++nidx)
  {
    if (layer == 0 && nidx > 0)
    {
      // Break on layer 0 after the first node since layer 0
      // contains only the root
      break;
    }

    // Skip non reachable states
    if (!pMDDGraph->isReachable(layer, nidx))
    {
      continue;
    }

    // Compute all next states from current node and filter only the "best" states
    std::vector<std::pair<MDDTDEdge*, bool>> newActiveEdgeList;

    // @note the original restricted algorithm, given layer i, first creates all 'k' states
    // for layer i+1; then it shrinks them to 'w' according to a selection heuristic.
    // In what follows, we create only the states that will be part of layer i+1.
    // In other words, instead of computing all 'k' states, the algorithm selects the 'w'
    // best states at each iteration and overrides them
    newActiveEdgeList = restrictNextLayerStatesFromNode(layer, nidx,
                                                        var->getLowerBound(),
                                                        var->getUpperBound(),
                                                        numGenStates);

    // All edges have "nidx" as tail node and lead to a node on layer "lidx+1"
    for (const auto& edgePair : newActiveEdgeList)
    {
      assert(edgePair.first->tail == nidx);

      // This layer has at least one successor on the next layer
      layerHasSuccessors = true;
      if (edgePair.second)
      {
        // Deactivate all current layers and active given layer
        for (auto edge : pMDDGraph->getEdgeOnHeadMutable(layer, edgePair.first->head))
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

  // The MDD is NOT exact if a layer exceeds the width
  if (numGenStates > pMDDGraph->getMaxWidth())
  {
    pIsExactMDD = false;
  }

  return true;
}

bool TDCompiler::buildRelaxedMDD(uint32_t layer, bool& setEdgesForCutset)
{
  // Get the variable associated with the current layer.
  // @note the variable is on the tail of the edge on the current layer
  auto var = pMDDGraph->getVariablePerLayer(layer);

  // Go over each node on the current layer and calculate next layer's nodes.
  // @note the relaxed MDD merges nodes to reduce the layer to the maximum allowed width.
  // The merge procedure needs the complete view of the nodes to merge.
  // This implies that all "next" states need to be created before merging them
  std::vector<DPState::UPtr> nextStatesList;
  nextStatesList.reserve(2 * pMDDGraph->getMaxWidth());

  // Step 0: prepare the data-structure to collect the values of the edges
  //         arriving at each node
  bool cutsetEdgeFound{false};
  spp::sparse_hash_map<uint32_t, std::vector<double>> edgeCostMap;
  spp::sparse_hash_map<uint32_t, std::vector<uint32_t>> edgeTailMap;
  spp::sparse_hash_map<uint32_t, std::vector<int64_t>> edgeValueMap;
  for (uint32_t nidx{0}; nidx < pMDDGraph->getMaxWidth(); ++nidx)
  {
    // Go over each node of the current layer and get the list of states
    // reachable from the current node
    if (layer == 0 && nidx > 0)
    {
      // Break on layer 0 after the first node since layer 0
      // contains only the root
      break;
    }

    // Skip non reachable states
    if (!pMDDGraph->isReachable(layer, nidx))
    {
      continue;
    }

    // Step 1: get the list of best next states reachable from the current state
    // according to the heuristic implemented in the DP model and initialize
    // the data structures
    auto currentState = pMDDGraph->getNodeState(layer, nidx);
    for (auto& state : currentState->nextStateList(var->getLowerBound(),
                                                   var->getUpperBound(),
                                                   getIncumbent()))
    {
      const auto nodeId = state->getUniqueId();


      edgeCostMap[nodeId] = {currentState->getCostOnValue(state->cumulativePath().back())};
      edgeValueMap[nodeId] = {state->cumulativePath().back()};
      edgeTailMap[state->getUniqueId()] = {nidx};
      nextStatesList.push_back(std::move(state));
    }
  }

  // Check if the layer has successors
  if (nextStatesList.empty())
  {
    // No successors, the MDD cannot be built
    return false;
  }

  // Get the constraint modeling the problem
  auto con = pProblem->getConstraints().at(0).get();


  // Step 2: unify nodes that are equal,
  // i.e., given n states, there may be subsets of the n states of states
  // that are equal to each other.
  // These states would be equal also on the exact MDD.
  // Merge these states first
  bool keepRepresentativeOnly{true};
  const auto equalStatesList = con->calculateEqualStates(nextStatesList);
  relaxNextNodesList(nextStatesList, equalStatesList,
                     edgeTailMap, edgeValueMap, edgeCostMap,
                     layer,
                     keepRepresentativeOnly);

  // Step 3: merge nodes according to the DP state equivalence function
  keepRepresentativeOnly = false;
  while(nextStatesList.size() > pMDDGraph->getMaxWidth())
  {
    const auto mergeStatesList = con->calculateMergeStates(
            nextStatesList, pMDDGraph->getMaxWidth());
    relaxNextNodesList(nextStatesList, mergeStatesList,
                       edgeTailMap, edgeValueMap, edgeCostMap,
                       layer,
                       keepRepresentativeOnly);
  }

  // Replace default states with the merged states
  const auto nextLayer{layer + 1};
  uint32_t nextDefaultNodeIdx{0};
  auto nextLayerStates = pMDDGraph->getStateListMutable(nextLayer);
  for (auto& node : nextStatesList)
  {
    // Scan all the edges incoming to the current "next" node,
    // activate them and set the costs and values on them
    spp::sparse_hash_map<uint32_t, spp::sparse_hash_map<int64_t, double>> valuesOnEdge;
    for (auto edgeTail : edgeTailMap.at(node->getUniqueId()))
    {
      // Set the lower cost as well as the value leading to the lower cost
      assert(edgeCostMap.at(node->getUniqueId()).size() ==
              edgeValueMap.at(node->getUniqueId()).size());

      const auto& allValuesList = edgeValueMap.at(node->getUniqueId());
      for (uint32_t idx{0}; idx < static_cast<uint32_t>(allValuesList.size()); ++idx)
      {
        const auto currVal = allValuesList.at(idx);
        if (valuesOnEdge[edgeTail].find(currVal) == valuesOnEdge[edgeTail].end())
        {
          valuesOnEdge[edgeTail][currVal] = edgeCostMap.at(node->getUniqueId()).at(idx);
        }
        else
        {
          valuesOnEdge[edgeTail][currVal] = std::min<double>(
                  valuesOnEdge[edgeTail][currVal], edgeCostMap.at(node->getUniqueId()).at(idx));
        }
      }
    }

    for (const auto& it : valuesOnEdge)
    {
      auto edge = pMDDGraph->getEdgeMutable(layer, it.first, nextDefaultNodeIdx);
      const auto& valueMap = it.second;

      // Find the minimum value
      double bestCost{std::numeric_limits<double>::max()};
      int64_t bestVal{std::numeric_limits<int64_t>::max()};
      bool setVal{false};

      if (valueMap.size() > 1)
      {
        cutsetEdgeFound = true;
      }

      for (const auto& valIt : valueMap)
      {
        if (valIt.second < bestCost)
        {
          bestCost = valIt.second;
          bestVal = valIt.first;
        }

        // Set all the values on the edge if the cutset has not being set yet
        if (!setEdgesForCutset)
        {
          if (!setVal)
          {
            edge->valuesList[0] = valIt.first;
            setVal = true;
          }
          else
          {
            edge->valuesList.push_back(valIt.first);
          }
        }
      }

      // @note in general, set only one value per edge even if
      // the edge is a parallel edge.
      // The value to set is the one that leads to a lower cost.
      // Setting one single value means that some sub-MDD are lost
      // when finding the cutset.
      // To avoid loosing sub-MDDs, the idea is to "expand",
      // i.e., create only one, the first, parallel edge since from
      // that edge, all other sub-MDDs can be re-built
      edge->costList[0] = bestCost;
      if (setEdgesForCutset)
      {
        // Set one value only if the cutset edge has been set
        edge->valuesList[0] = bestVal;
      }
      edge->isActive = true;
    }

    pMDDGraph->replaceState(nextLayer, nextDefaultNodeIdx, std::move(node));
    pMDDGraph->getNodeState(nextLayer, nextDefaultNodeIdx)->setNonDefaultState();
    if (layer < (pMDDGraph->getNumLayers() - 1))
    {
      // @note last layer always has one node,
      // i.e., the terminal node
      ++nextDefaultNodeIdx;
    }
  }

  if(cutsetEdgeFound)
  {
    // One cutset edge has been found and set
    setEdgesForCutset = true;
  }

  return true;
}

void TDCompiler::relaxNextNodesList(
        std::vector<DPState::UPtr>& nextStatesList,
        const std::vector<std::vector<uint32_t>>& mergeList,
        spp::sparse_hash_map<uint32_t, std::vector<uint32_t>>& edgeTailMap,
        spp::sparse_hash_map<uint32_t, std::vector<int64_t>>& edgeValueMap,
        spp::sparse_hash_map<uint32_t, std::vector<double>>& edgeCostMap,
        uint32_t layer,
        bool keepRepresentativeOnly)
{
  for (const auto& sublist : mergeList)
  {
    // Keep only one representative (i.e., the first)
    // for each equal state but keep track of the paths leading to those states
    auto representativeNode = nextStatesList.at(sublist.at(0)).get();
    const auto nodeId = representativeNode->getUniqueId();
    for (uint32_t idx{1}; idx < static_cast<uint32_t>(sublist.size()); ++idx)
    {
      assert(edgeTailMap.at(nextStatesList.at(sublist.at(idx))->getUniqueId()).size() == 1);

      // Get the value leading to that "equal" state and store the value and its cost
      auto nodeToMerge = nextStatesList.at(sublist.at(idx)).get();
      const auto nodeToMergeId = nodeToMerge->getUniqueId();
      const auto val = edgeValueMap.at(nodeToMergeId).front();
      const auto cost = edgeCostMap.at(nodeToMergeId).front();
      const auto tail = edgeTailMap.at(nodeToMergeId).front();

      if (!keepRepresentativeOnly)
      {
        // Since this is not replacing nodes but merging,
        // add cost, value, and the tail
        edgeCostMap[nodeId].push_back(cost);
        edgeValueMap[nodeId].push_back(val);
        edgeTailMap[nodeId].push_back(tail);

        // Nodes should be merged rather than keeping only the representative node
        representativeNode->mergeState(nodeToMerge);
        representativeNode->setNonDefaultState();
        representativeNode->setExact(false);
      }

      // Remove the merged state
      nextStatesList[sublist.at(idx)].reset();
    }
  }

  // Remove all nullptr states
  nextStatesList.erase(std::remove_if(
          nextStatesList.begin(), nextStatesList.end(),
          [](auto const& ptr){ return ptr == nullptr; }), nextStatesList.end());
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

      // Compute all next states from current node and merge or filter the states
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
  auto fromState = pMDDGraph->getNodeState(currLayer, currNode);
  assert(fromState != nullptr);

  // Get the list of next states to override (since only "width" states are allowed)
  const auto nextLayer = currLayer + 1;
  TopDownMDD::StateList* nextStateList = pMDDGraph->getStateListMutable(nextLayer);

  // Get the list of best "width" next states reachable from the current state
  // according to the heuristic implemented in the DP model
  auto stateList = fromState->nextStateList(lb, ub, getIncumbent());
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
    edge->costList[0] = fromState->getCostOnValue(val);
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
          nextStateList->at(idx)->forceCumulativePath(currState->cumulativePath());
        }

        // Merge a new edge
        auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, idx);
        assert(edge != nullptr);

        if (!edge->hasValueSet())
        {
          edge->valuesList[0] = currState->cumulativePath().back();
          edge->costList[0] = fromState->getCostOnValue(edge->valuesList[0]);
        }
        else
        {
          edge->valuesList.push_back(currState->cumulativePath().back());
          edge->costList.push_back(fromState->getCostOnValue(edge->valuesList.back()));
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
        const auto cost = fromState->getCostOnValue(val);
        pMDDGraph->replaceState(nextLayer, idx, std::move(stateList[repPtr]));

        // Activate a new edge
        auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, idx);
        edge->valuesList[0] = val;
        edge->costList[0] = cost;
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
  auto fromState = pMDDGraph->getNodeState(currLayer, currNode);
  assert(fromState != nullptr);

  // Get the list of next states to override (since only "width" states are allowed)
  const auto nextLayer = currLayer + 1;
  TopDownMDD::StateList* nextStateList = pMDDGraph->getStateListMutable(nextLayer);

  // Get the list of best "width" next states reachable from the current state
  // according to the heuristic implemented in the DP model
  auto stateList = fromState->nextStateList(lb, ub,  getIncumbent());

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
    edge->costList[0] = fromState->getCostOnValue(val);
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
          nextStateList->at(idx)->forceCumulativePath(currState->cumulativePath());
        }

        // Merge a new edge
        auto edge = pMDDGraph->getEdgeMutable(currLayer, currNode, idx);
        assert(edge != nullptr);

        if (!edge->hasValueSet())
        {
          edge->valuesList[0] = currState->cumulativePath().back();
          edge->costList[0] = fromState->getCostOnValue(edge->valuesList[0]);
        }
        else
        {
          edge->valuesList.push_back(currState->cumulativePath().back());
          edge->costList.push_back(fromState->getCostOnValue(edge->valuesList.back()));
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
      edge->costList[0] = fromState->getCostOnValue(val);
    }
    else
    {
      edge->valuesList.push_back(val);
      edge->costList.push_back(fromState->getCostOnValue(val));
    }
    newConnections.emplace_back(edge, false);

    // Continue with next state
    ++repPtr;
  }

  return newConnections;
}

}  // namespace mdd
