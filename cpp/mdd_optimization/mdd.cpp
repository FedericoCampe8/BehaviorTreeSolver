#include "mdd_optimization/mdd.hpp"

#include <sparsepp/spp.h>

#include <algorithm>  // for std::find
#include <cassert>
#include <cstdlib>    // for std::rand
#include <fstream>
#include <iostream>
#include <queue>      // for std::priority_queue
#include <stack>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::pair

#include <sparsepp/spp.h>

#include "tools/timer.hpp"

// #define DEBUG

namespace {
constexpr uint32_t kLayerZero{0};

class CompareNodesOnCost
{
public:
  CompareNodesOnCost(){}
  bool operator() (const mdd::Node* lhs, const mdd::Node* rhs) const
  {
    return (lhs->getDPState()->cumulativeCost() > lhs->getDPState()->cumulativeCost());
  }
};

}  // namespace

namespace mdd {

MDD::MDD(MDDProblem::SPtr problem, int32_t width)
: pMaxWidth(width),
  pProblem(problem),
  pArena(std::make_unique<Arena>())
{
  if (problem == nullptr)
  {
    throw std::invalid_argument("MDD - empty pointer to the problem");
  }

  if (width < 1)
  {
    throw std::invalid_argument("MDD - invalid width size");
  }

  // TODO: max width not implemented in constraints

  // Resize the number of layers of this MDD to have one layer per variable in the problem.
  // Note: there is one more corresponding to the terminal node layer
  pNodesPerLayer.resize(pProblem->getVariables().size() + 1);
}

void MDD::enforceConstraints(MDDConstructionAlgorithm algorithmType)
{
  if (algorithmType == MDDConstructionAlgorithm::Separation)
  {
    // Start from the relaxed MDD
    auto relaxedMDD = buildRelaxedMDD();
    runSeparationProcedure(relaxedMDD);
  }
  else if(algorithmType == MDDConstructionAlgorithm::Filtering)
  {
    // Start from the relaxed MDD
    auto relaxedMDD = buildRelaxedMDD();
    runFilteringProcedure(relaxedMDD);
  }
  else if(algorithmType == MDDConstructionAlgorithm::SeparationWithIncrementalRefinement)
  {
    // Start from the relaxed MDD
    auto relaxedMDD = buildRelaxedMDD();
    runSeparationAndRefinementProcedure(relaxedMDD);
  }
  else if(algorithmType == MDDConstructionAlgorithm::TopDown)
  {
    // Start from the root node
    auto rootNode = buildRootMDD();
    pRootNode->initializeNodeDomain();
    runTopDownProcedure(rootNode, false);
  }
  else if(algorithmType == MDDConstructionAlgorithm::RestrictedTopDown)
  {
    // Start from the root node
    auto rootNode = buildRootMDD();
    pRootNode->initializeNodeDomain();
    runTopDownProcedure(rootNode, true);
  }
  else
  {
    throw std::runtime_error("MDD - enforceConstraints: MDD compilation type not supported");
  }
}

Node* MDD::buildRootMDD()
{
  // Build the root node
  pNodesPerLayer.at(kLayerZero).clear();
  pNodesPerLayer.at(kLayerZero).push_back(
          pArena->buildNode(kLayerZero, pProblem->getVariables().at(kLayerZero).get()));

  pRootNode = pNodesPerLayer.at(kLayerZero).back();
  return pRootNode;
}

void MDD::revertMDD()
{
  // Swap all edges in the arena
  const auto& edges = pArena->getEdgePool();
  for (auto& edgeIt : edges)
  {
    edgeIt.second->reverseEdge();
  }

  // Swap the layers of the mdd top to bottom
  MDDLayersList reversedMDD = pNodesPerLayer;
  pNodesPerLayer.clear();

  uint32_t layer{kLayerZero};
  while(!reversedMDD.empty())
  {
    pNodesPerLayer.push_back(reversedMDD.back());
    std::reverse(std::begin(pNodesPerLayer.back()), std::end(pNodesPerLayer.back()));
    for (auto node : pNodesPerLayer.back())
    {
      assert(pArena->containsNode(node->getUniqueId()));
      assert(!node->hasDefaultDPState());
      node->resetLayer(layer);
    }
    reversedMDD.pop_back();
    ++layer;
  }
  pRootNode = pNodesPerLayer.at(0).at(0);
}

Node* MDD::buildRelaxedMDD()
{
  // Build the root node
  buildRootMDD();

  Node* currNode = pRootNode;
  const auto totLayers = static_cast<int>(pProblem->getVariables().size());
  for (int idx{0}; idx < totLayers; ++idx)
  {
    auto nextNode = expandNode(currNode);
    currNode = nextNode;
    pNodesPerLayer.at(currNode->getLayer()).push_back(currNode);

    // Set the terminal node (updated at each cycle until the last one)
    pTerminalNode = currNode;
  };

  return pRootNode;
}

Node* MDD::expandNode(Node* node)
{
  // Get the values to pair with the incoming edge on the new node to create.
  // The values are given by the variable paired with the current level
  assert(node != nullptr);
  auto currLayer = node->getLayer();
  auto var = pProblem->getVariables().at(currLayer).get();

  Variable* nextVar{nullptr};
  if (currLayer + 1 < static_cast<uint32_t>(pProblem->getVariables().size()))
  {
    // Notice that the last node, i.e., the terminal node, doesn't have any
    // variable associated with it
    nextVar = pProblem->getVariables().at(currLayer+1).get();
  }

  auto nextNode = pArena->buildNode(currLayer + 1, nextVar);

  // Create an edge connecting the two nodes.
  // Notice that the Edge constructor will automatically link the edge to the
  // tail and head nodes.
  // TODO avoid creating one edge per domain element but use lower and upper bounds
  //      on one single edge
  auto lbValue = node->getVariable()->getLowerBound();
  for (; lbValue <= node->getVariable()->getUpperBound(); ++lbValue)
  {
    pArena->buildEdge(node, nextNode, lbValue, lbValue);
  }

  // Return next node
  return nextNode;
}

void MDD::runSeparationProcedure(Node* root)
{
  for (auto& con : pProblem->getConstraints())
  {
    runSeparationProcedureOnConstraint(root, con.get());

    // Set single node as bottom node:
    // merge all nodes in the bottom layer
    int lastNodeLayer = static_cast<int>(pNodesPerLayer.size() - 1);
    int numNodesInLastLayer = static_cast<int>(pNodesPerLayer.at(lastNodeLayer).size());
    if (numNodesInLastLayer > 1)
    {
      auto refNode = pNodesPerLayer.at(lastNodeLayer).at(0);
      for (int nidx{1}; nidx < numNodesInLastLayer; ++nidx)
      {
        // Move all incoming layer to the reference node
        auto currNode = pNodesPerLayer.at(lastNodeLayer).at(nidx);
        for (auto edge : currNode->getInEdges())
        {
          edge->setHead(refNode);
        }
        pArena->deleteNode(currNode->getUniqueId());
      }

      // Pop back the nodes from the last layer
      while(pNodesPerLayer.at(lastNodeLayer).size() > 1)
      {
        pNodesPerLayer[lastNodeLayer].pop_back();
      }
    }

    if (con->runsBottomUp())
    {
      const bool bottomUp{true};

      // Revert the MDD
      revertMDD();

      // Run separation on reverted MDD
      con->setForBottomUpFiltering();
      runSeparationProcedureOnConstraint(pRootNode, con.get(), bottomUp);

      // Reset MDD
      revertMDD();

      // Reset constraint for top-down approach
      con->setForTopDownFiltering();
    }
  }
}

void MDD::runSeparationProcedureOnConstraint(Node* root, MDDConstraint* con, bool bottomUp)
{
  // Step 1: set default state on each node
  if (!bottomUp)
  {
    // Note: on bottom-up separation, the nodes' states must be preserved
    for (auto& layer : pNodesPerLayer)
    {
      for (auto node : layer)
      {
        if (!node->hasDefaultDPState())
        {
          node->setDefaultDPState();
        }
      }
    }
  }

  // Step 2: set the initial constraint DP state on the root
  root->resetDPState(con->getInitialDPState());
  if (bottomUp)
  {
    root->getDPState()->setStateForTopDownFiltering(false);
  }

  // Step 3: for each layer, for each node, for each arc,
  //         compute the next DP state and split nodes accordingly
  auto totLayers = static_cast<uint32_t>(pProblem->getVariables().size());
  spp::sparse_hash_set<uint32_t> bottomUpNodesStateVisitMap;
  std::vector<std::pair<DPState*, Node*>> newDPStates;
  newDPStates.reserve(pMaxWidth);
  for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
  {
#ifdef DEBUG
    std::cout << "Processed layer: " << layerIdx << std::endl;
#endif

    // Store the states created at each layer
    newDPStates.clear();
    for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
    {
      auto node = pNodesPerLayer.at(layerIdx).at(nodeIdx);
      auto currDPState = node->getDPState();

#ifdef DEBUG
      std::cout << "Processed node index: " << nodeIdx << " with state " <<
              currDPState->toString() << " and num out edges " <<
              node->getOutEdges().size() << std::endl;
#endif
      // Get all outgoing edges and, for each edge, check next state
      // w.r.t. the edge's value.
      // Note: create a copy of the edge list since this will be modifies
      Node::EdgeList edgeList = node->getOutEdges();
      for (auto edge : edgeList)
      {
        // Notice that if the edge is a parallel edge,
        // there will be multiple values to process.
        // However, in current implementation (9/20/2020) we assume
        // that every edge has one single element on it.
        // TODO remove the above assumption
        assert(edge->getDomainSize() == 1);
        assert(edge->getHead() != nullptr);
        assert(edge->getTail() != nullptr);
        auto edgeValue = edge->getDomainLowerBound();

#ifdef DEBUG
        std::cout << "Consider the edge from: " << edge->getTail()->getDPState()->toString()
                << " to " << edge->getHead()->getDPState()->toString() <<
                " with value " << edge->getValue() << std::endl;
#endif
        // Calculate the next DP state, i.e., the state reachable from the current one
        // by applying the given edge/arc value.
        // Note: the bottom up pass needs the information of the next state to go to
        DPState* inState = bottomUp ? edge->getHead()->getDPState() : nullptr;
        assert(pArena->containsEdge(edge->getUniqueId()));
        assert(pArena->containsNode(edge->getHead()->getUniqueId()));
        if (edge->getHead()->hasDefaultDPState())
        {
          inState = nullptr;
        }

        auto newDPState = currDPState->next(edgeValue, inState);
        if (bottomUp && (bottomUpNodesStateVisitMap.find(edge->getHead()->getUniqueId()) ==
                bottomUpNodesStateVisitMap.end()))
        {
          // Reset the state for the head since it won't be used but the first time.
          // Note: this reset is done only the first time this node is touched and
          // updated. This is why there is a set "bottomUpNodesStateVisitMap"
          edge->getHead()->setDefaultDPState();
          bottomUpNodesStateVisitMap.insert(edge->getHead()->getUniqueId());
        }

        if (newDPState->isInfeasible())
        {
          // If the new state is infeasible: remove this arc from tail and head nodes
          edge->removeEdgeFromNodes();
        }
        else if (edge->getHead()->hasDefaultDPState())
        {
          // The head node has a default state: set this state as its new state
          // and continue
          edge->getHead()->resetDPState(newDPState);
          newDPStates.push_back({newDPState.get(), edge->getHead()});
        }
        else
        {
          // Check if another state similar to the current one has been
          // already created in the same layer
          Node* matchingNode{nullptr};
          for (const auto& state : newDPStates)
          {
            if (state.first->isEqual(newDPState.get()))
            {
              // A match is found
              matchingNode = state.second;
              break;
            }
          }

          if (matchingNode != nullptr)
          {
            // A match to an existing state is found.
            // Consider two cases:
            // a) there is already an edge connecting the two nodes: nothing to do
            // b) there is no edge connecting the two nodes: remove current edge,
            //    and create a new edge from the current node to the matching state
            assert(node != matchingNode);
            assert(node->getLayer() < matchingNode->getLayer());

            // Note: scan on the new node's edge list with all the most recent inserted edges
            bool foundCompatibleEdge{false};
            for (auto nodeTestEdge : node->getOutEdges())
            {
              if ((nodeTestEdge->getHead()->getUniqueId() == matchingNode->getUniqueId()) &&
                      nodeTestEdge->getDomainLowerBound() == edgeValue)
              {
                // Found the same edge: nothing to do
                foundCompatibleEdge = true;
                break;
              }
            }

            if (!foundCompatibleEdge)
            {
              // Remove current edge
              edge->removeEdgeFromNodes();

              // Create a new edge with matching nodes
              pArena->buildEdge(node, matchingNode, edgeValue, edgeValue);
            }
          }
          else
          {
            // No matching node, create, split the node
            // Head node already has a non-default state: split the node
            auto nextNode = edge->getHead();

            // Step I: remove the current edge from the two nodes
            edge->removeEdgeFromNodes();

            // Step II: create a new node with the new DP state
            auto nextNewNode = pArena->buildNode(nextNode->getLayer(), nextNode->getVariable());
            nextNewNode->resetDPState(newDPState);
            newDPStates.push_back({newDPState.get(), nextNewNode});

            // Step III: add arc from node to nextNewNode and value the value that led to newDPState
            assert(node != nextNewNode);
            assert(node->getLayer() < nextNewNode->getLayer());
            auto newEdge = pArena->buildEdge(node, nextNewNode, edgeValue, edgeValue);

            // Step IV: copy outgoing arcs of nextNode on nextNewNode
            for (auto nextEdge : nextNode->getOutEdges())
            {
              assert(nextNewNode != nextEdge->getHead());
              pArena->buildEdge(nextNewNode, nextEdge->getHead(), nextEdge->getDomainLowerBound(),
                                nextEdge->getDomainUpperBound());
            }

            // Step V: add this node to the next layer
            pNodesPerLayer.at(nextNode->getLayer()).push_back(nextNewNode);
          }
        }
      }  // for each out edge
    }  // for each node in layer
  }  // for each layer

#ifdef DEBUG
  for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
  {
    std::cout << "Processing layer " << layerIdx << " with nodes " <<
            pNodesPerLayer.at(layerIdx).size() << std::endl;
    for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
    {
      auto node = pNodesPerLayer.at(layerIdx).at(nodeIdx);
      std::cout << "node " << node->getDPState()->toString() << std::endl;
      std::cout << "In " << node->getInEdges().size() << ": " << std::endl;
      for (auto edge : node->getInEdges())
      {
        std::cout << edge->getHead()->getDPState()->toString() << " ";
      }
      std::cout << "\nout " << node->getOutEdges().size() << ": " << std::endl;
      for (auto edge : node->getOutEdges())
      {
        std::cout << edge->getValue() << ": " << edge->getHead()->getDPState()->toString() << " ";
      }
      std::cout << std::endl;
    }
  }
#endif
}

void MDD::runSeparationAndRefinementProcedure(Node* root)
{
  for (auto& con : pProblem->getConstraints())
  {
    runSeparationAndRefinementProcedureOnConstraint(root, con.get());
  }
}

void MDD::runSeparationAndRefinementProcedureOnConstraint(Node* root, MDDConstraint* con)
{
  // Step 1: set default state on each node
  for (auto& layer : pNodesPerLayer)
  {
    for (auto node : layer)
    {
      if (!node->hasDefaultDPState())
      {
        node->setDefaultDPState();
      }
    }
  }

  // Step 2: set the initial constraint DP state on the root
  root->resetDPState(con->getInitialDPState());

  // Step 3: for each layer, perform filtering and refinement
  auto totLayers = static_cast<uint32_t>(pProblem->getVariables().size());
  std::vector<std::pair<DPState*, Node*>> newDPStates;
  newDPStates.reserve(pMaxWidth);
  for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
  {
    // Store the states created at each layer
    newDPStates.clear();

    /******************
     *   FILTERING
     ******************/
    // For each node in the current layer, for each arc leaving the node,
    // check if the arc is leading to an invalid state.
    // If so, remove the arc
    for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
    {
      auto node = pNodesPerLayer.at(layerIdx).at(nodeIdx);
      auto currDPState = node->getDPState();

      // Get all outgoing edges and, for each edge, check next state
      // w.r.t. the edge's value.
      // Note: create a copy of the edge list since this will be modifies
      Node::EdgeList edgeList = node->getOutEdges();
      for (auto edge : edgeList)
      {
        // Notice that if the edge is a parallel edge,
        // there will be multiple values to process.
        // However, in current implementation (9/20/2020) we assume
        // that every edge has one single element on it.
        // TODO remove the above assumption
        assert(edge->getDomainSize() == 1);
        assert(edge->getHead() != nullptr);
        assert(edge->getTail() != nullptr);
        auto edgeValue = edge->getDomainLowerBound();

        // Calculate the next DP state, i.e., the state reachable from the current one
        // by applying the given edge/arc value
        auto newDPState = currDPState->next(edgeValue);
        if (newDPState->isInfeasible())
        {
          // If the new state is infeasible: remove this arc from tail and head nodes
          edge->removeEdgeFromNodes();
        }
      }  // for each out edge
    }  // for each node in layer

    /******************
     *   REFINEMENT
     ******************/
    // For each node in the current layer, for each arc leaving the node,
    // apply standard separation but up to a width limit
    for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
    {
      auto node = pNodesPerLayer.at(layerIdx).at(nodeIdx);
      auto currDPState = node->getDPState();

      // Get all outgoing edges and, for each edge, check next state
      // w.r.t. the edge's value.
      // Note: create a copy of the edge list since this will be modifies.
      // Note: thanks to the previous filtering step, all edges here are valid edges
      Node::EdgeList edgeList = node->getOutEdges();
      for (auto edge : edgeList)
      {
        // Notice that if the edge is a parallel edge,
        // there will be multiple values to process.
        // However, in current implementation (9/20/2020) we assume
        // that every edge has one single element on it.
        // TODO remove the above assumption
        assert(edge->getDomainSize() == 1);
        assert(edge->getHead() != nullptr);
        assert(edge->getTail() != nullptr);
        auto edgeValue = edge->getDomainLowerBound();

        // Calculate the next DP state, i.e., the state reachable from the current one
        // by applying the given edge/arc value
        auto newDPState = currDPState->next(edgeValue);
        assert(!newDPState->isInfeasible());

        if (edge->getHead()->hasDefaultDPState())
        {
          // The head node has a default state: set this state as its new state
          // and continue
          edge->getHead()->resetDPState(newDPState);
          newDPStates.push_back({newDPState.get(), edge->getHead()});
        }
        else if(pNodesPerLayer.at(edge->getHead()->getLayer()).size() < pMaxWidth)
        {
          // Check if another state similar to the current one has been
          // already created in the same layer
          Node* matchingNode{nullptr};
          for (const auto& state : newDPStates)
          {
            if (state.first->isEqual(newDPState.get()))
            {
              // A match is found
              matchingNode = state.second;
              break;
            }
          }

          if (matchingNode != nullptr)
          {
            // A match to an existing state is found.
            // Consider two cases:
            // a) there is already an edge connecting the two nodes: nothing to do
            // b) there is no edge connecting the two nodes: remove current edge,
            //    and create a new edge from the current node to the matching state
            assert(node != matchingNode);
            assert(node->getLayer() < matchingNode->getLayer());

            // Note: scan on the new node's edge list with all the most recent inserted edges
            bool foundCompatibleEdge{false};
            for (auto nodeTestEdge : node->getOutEdges())
            {
              if ((nodeTestEdge->getHead()->getUniqueId() == matchingNode->getUniqueId()) &&
                      nodeTestEdge->getDomainLowerBound() == edgeValue)
              {
                // Found the same edge: nothing to do
                foundCompatibleEdge = true;
                break;
              }
            }

            if (!foundCompatibleEdge)
            {
              // Remove current edge
              edge->removeEdgeFromNodes();

              // Create a new edge with matching nodes
              pArena->buildEdge(node, matchingNode, edgeValue, edgeValue);
            }
          }
          else
          {
            // No matching node, create, split the node
            // Head node already has a non-default state: split the node
            auto nextNode = edge->getHead();

            // Step I: remove the current edge from the two nodes
            edge->removeEdgeFromNodes();

            // Step II: create a new node with the new DP state
            auto nextNewNode = pArena->buildNode(nextNode->getLayer(), nextNode->getVariable());
            nextNewNode->resetDPState(newDPState);
            newDPStates.push_back({newDPState.get(), nextNewNode});

            // Step III: add arc from node to nextNewNode and value the value that led to newDPState
            assert(node != nextNewNode);
            assert(node->getLayer() < nextNewNode->getLayer());
            auto newEdge = pArena->buildEdge(node, nextNewNode, edgeValue, edgeValue);

            // Step IV: copy outgoing arcs of nextNode on nextNewNode
            for (auto nextEdge : nextNode->getOutEdges())
            {
              assert(nextNewNode != nextEdge->getHead());
              pArena->buildEdge(nextNewNode, nextEdge->getHead(), nextEdge->getDomainLowerBound(),
                                nextEdge->getDomainUpperBound());
            }

            // Step V: add this node to the next layer
            pNodesPerLayer.at(nextNode->getLayer()).push_back(nextNewNode);
          }
        }
        else
        {
          // A max limit is reached and the next state is valid:
          // update the head of the current edge with the next state "newDPState".
          // A way to update the head of the current edge is to merge it with
          // the next state "newDPState".
          // Before blindly merge the state, check if "newDPState" is equal to one of the states
          // newly added to the current layer.
          // If so, don't merge but re-direct the current edge to point to that state.
          assert(!edge->getHead()->hasDefaultDPState());
          Node* matchingNode{nullptr};
          for (const auto& state : newDPStates)
          {
            if (state.first->isEqual(newDPState.get()))
            {
              // A match is found
              matchingNode = state.second;
              break;
            }
          }

          if (matchingNode != nullptr)
          {
            // A match to an existing state is found.
            // Consider two cases:
            // a) there is already an edge connecting the two nodes: nothing to do
            // b) there is no edge connecting the two nodes: remove current edge,
            //    and create a new edge from the current node to the matching state
            assert(node != matchingNode);
            assert(node->getLayer() < matchingNode->getLayer());

            // Note: scan on the new node's edge list with all the most recent inserted edges
            bool foundCompatibleEdge{false};
            for (auto nodeTestEdge : node->getOutEdges())
            {
              if ((nodeTestEdge->getHead()->getUniqueId() == matchingNode->getUniqueId()) &&
                      nodeTestEdge->getDomainLowerBound() == edgeValue)
              {
                // Found the same edge: nothing to do
                foundCompatibleEdge = true;
                break;
              }
            }

            if (!foundCompatibleEdge)
            {
              // Remove current edge
              edge->removeEdgeFromNodes();

              // Create a new edge with matching nodes
              pArena->buildEdge(node, matchingNode, edgeValue, edgeValue);
            }
          }
          else
          {
            // No similar state is found: merge the new state to the head of the current edge
            edge->getHead()->getDPState()->mergeState(newDPState.get());
          }
        }
      }  // for each out edge
    }  // for each node in layer
  }  // for each layer
}

void MDD::runTopDownProcedure(Node* node, bool isRestricted)
{
  // srand( (unsigned)time(NULL) );
  if (node == nullptr)
  {
    throw std::runtime_error("MDD - runTopDownProcedure: empty pointer to node");
  }

  // The Top-Down procedure assumes that there is ONLY ONE DP model
  // encoded as a constraint that represents the problem
  const auto& conList = pProblem->getConstraints();
  if (conList.size() != 1)
  {
    throw std::runtime_error("MDD - runTopDownProcedure: the model should be defined by "
            "one and only one Dynamic Programming model");
  }
  auto conDPModel = conList.at(0);

  // Set the first state
  pNodesPerLayer.at(kLayerZero).at(0)->resetDPState(conDPModel->getInitialDPState());

  // Get the total number of layers
  const auto totLayers = static_cast<int>(pProblem->getVariables().size());

  // Keep track of the discarded nodes per layer
  std::vector<std::priority_queue<Node*, std::vector<Node*>, CompareNodesOnCost>> nodePool;
  nodePool.resize(totLayers);

  // For all layers
  std::vector<std::pair<DPState*, Node*>> newDPStates;
  newDPStates.reserve(std::min(32, pMaxWidth));
  for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
  {
    // Reset the list of new states
    newDPStates.clear();

    // Apply merging procedure for relaxed MDDs
    while((!isRestricted) && (layerIdx > 0) && (pNodesPerLayer.at(layerIdx).size() > pMaxWidth))
    {
      // Step I: select a subset of nodes to merge
      auto subsetNodes = conDPModel->mergeNodeSelect(layerIdx, getMDDRepresentation());
      assert(subsetNodes.size() > 1);

      // Step II: remove the selected nodes from the MDD
      std::vector<Edge*> edgesToRedirect;
      spp::sparse_hash_set<uint32_t> mergingNodes;
      auto& currentLevel = pNodesPerLayer[layerIdx];
      for (auto node : subsetNodes)
      {
        // Check that there are no duplicate nodes
        if (mergingNodes.find(node->getUniqueId()) != mergingNodes.end())
        {
          throw std::runtime_error("MDD - runTopDownProcedure: "
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
      }

      // Step III: Merge all the nodes into one and get their representative
      auto newMergedNode = conDPModel->mergeNodes(subsetNodes, pArena.get());

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

    // Apply filtering procedure for restricted MDDs
    while(isRestricted && (pMaxWidth > 1) && (pNodesPerLayer.at(layerIdx).size() > pMaxWidth))
    {
      // Simply remove nodes
      //auto randPos = std::rand() % (pNodesPerLayer.at(layerIdx).size());
      //auto lastNode = pNodesPerLayer.at(layerIdx).at(randPos);
      //pNodesPerLayer.at(layerIdx).erase(pNodesPerLayer.at(layerIdx).begin() + randPos);

      // Pick and remove the most expensive node
      int nodeToRemoveIdx{0};
      double worstCost{0};
      for (int nidx{0}; nidx < static_cast<int>(pNodesPerLayer.at(layerIdx).size()); ++nidx)
      {
        auto currNode = pNodesPerLayer.at(layerIdx).at(nidx);
        assert((currNode->getInEdges()).size() == 1);

        const auto currCost = currNode->getDPState()->cumulativeCost();
        if (currCost > worstCost)
        {
          worstCost = currCost;
          nodeToRemoveIdx = nidx;
        }
      }

      auto lastNode = pNodesPerLayer.at(layerIdx).at(nodeToRemoveIdx);
      pNodesPerLayer.at(layerIdx).erase(pNodesPerLayer.at(layerIdx).begin() + nodeToRemoveIdx);

      //auto lastNode = pNodesPerLayer.at(layerIdx).back();
      //pNodesPerLayer.at(layerIdx).pop_back();

      // Note: remove all the connected edges as well
      assert(lastNode->getOutEdges().empty());
      auto allEdges = lastNode->getInEdges();
      for (auto inEdgeToRemove : allEdges)
      {
        pArena->deleteEdge(inEdgeToRemove->getUniqueId());
      }

      // Store the node to be used later on during branch and bound
      nodePool.at(lastNode->getLayer()).push(lastNode);
      //pArena->deleteNode(lastNode->getUniqueId());
    }

    // For all nodes per layer
    for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
    {
      // For all values of the domain of the current layer
      auto currNode = pNodesPerLayer.at(layerIdx).at(nodeIdx);
      const auto& currDomain = currNode->getValues();
      for (auto val : currDomain)
      {
        // Calculate the DP state w.r.t. the given domain value
        auto nextDPState = currNode->getDPState()->next(val);
        if (nextDPState->isInfeasible())
        {
          // Continue with next value if this current value leads to an infeasible state
          continue;
        }

        // Check if the new DP state matches a DP state just created
        Node* matchingNode{nullptr};
        for (const auto& state : newDPStates)
        {
          break;
          if (state.first->isEqual(nextDPState.get()))
          {
            // A match is found
            matchingNode = state.second;
            break;
          }
        }

        if (matchingNode != nullptr)
        {
          // Found a match, re-use the same state
          // Create an edge connecting the current node and the next node
          pArena->buildEdge(currNode, matchingNode, val, val);
        }
        else
        {
          // Create a new node for the current state.
          // The new node should own the variable on next layer
          const auto nextLayer = currNode->getLayer() + 1;
          auto nextVar = nextLayer < (pProblem->getVariables()).size() ?
                  pProblem->getVariables().at(nextLayer).get() :
                  nullptr;
          auto nextNode = pArena->buildNode(currNode->getLayer() + 1, nextVar);
          if (nextVar != nullptr)
          {
            nextNode->initializeNodeDomain();
          }

          // Set the new DP state on the new node
          nextNode->resetDPState(nextDPState);

          // Create an edge connecting the current node and the next node
          pArena->buildEdge(currNode, nextNode, val, val);

          // Add the node to the next layer
          pNodesPerLayer.at(nextLayer).push_back(nextNode);

          // Store the current state to see if it can be re-used later
          newDPStates.push_back({nextDPState.get(), nextNode});
        }
      }
    }  // for each node
  }  // for each layer

  // Check min/max on node pool
  double minCost{std::numeric_limits<double>::max()};
  double maxCost{std::numeric_limits<double>::lowest()};
  for (auto& q : nodePool)
  {
    while(!q.empty())
    {
      auto topNode = q.top();
      const auto val = topNode->getDPState()->cumulativeCost();
      pArena->deleteNode(topNode->getUniqueId());
      q.pop();
      if (val < minCost) minCost = val;
      if (val > maxCost) maxCost = val;
    }
  }
  std::cout << "Min cost " << minCost << " Max cost " << maxCost << std::endl;

}

void MDD::runFilteringProcedure(Node* node)
{
  // Initialize all node domains
  auto totLayers = static_cast<uint32_t>(pProblem->getVariables().size());
  for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
  {
    for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
    {
      auto node = pNodesPerLayer.at(layerIdx).at(nodeIdx);
      node->initializeNodeDomain();
    }
  }

  // Enforce all constraints
  std::vector<Node*> newNodesList;
  for (auto& con : pProblem->getConstraints())
  {
    con->enforceConstraint(pArena.get(), pNodesPerLayer, newNodesList);
  }
}

std::vector<Edge*> MDD::maximize()
{
  std::stack<Node*> nodesToExpand;
  spp::sparse_hash_set<uint32_t> visited;

  assert(pRootNode != nullptr);
  pRootNode->setOptimizationValue(0.0);

  nodesToExpand.push(pRootNode);
  while(!nodesToExpand.empty())
  {
    auto currentNode = nodesToExpand.top();
    nodesToExpand.pop();

    // Many edges can point to the same node
    for (auto edge : currentNode->getOutEdges())
    {
      // Get the next node and check that it is visited only once
      auto nextNode = edge->getHead();
      bool nodeFound = (visited.find(nextNode->getUniqueId()) != visited.end());
      if (!nodeFound)
      {
          visited.insert(nextNode->getUniqueId());
      }

      if (!nodeFound)
      {
        // If node does not exists, mark it as node to expand
        // std::cout << "Add node with " << nextNode->getDPState()->toString() <<
        //        " from edge " << edge->getValue() << std::endl;
        nodesToExpand.push(nextNode);
        nextNode->setOptimizationValue(
                currentNode->getOptimizationValue() +
                currentNode->getDPState()->cost(edge->getValue(), edge->getTail()->getDPState()));
        nextNode->setSelectedEdge(edge);
      }
      else
      {
        // If it does, the node has been visited already,
        // update its partial solution choosing the maximum value
        double candidateValue = currentNode->getOptimizationValue() +
                currentNode->getDPState()->cost(edge->getValue(), edge->getTail()->getDPState());
        if (candidateValue > nextNode->getOptimizationValue())
        {
          nextNode->setOptimizationValue(candidateValue);
          nextNode->setSelectedEdge(edge);
        }
      }
    }
  }  // while

  std::vector<Edge*> solution;

  // Start with leaf and trace backwards
  auto node = pNodesPerLayer.at(pNodesPerLayer.size()- 1).at(0);
  while (node->getSelectedEdge() != nullptr)
  {
      solution.push_back(node->getSelectedEdge());
      node = node->getSelectedEdge()->getTail();
  }

  return solution;
}

std::vector<Edge*> MDD::minimize()
{
  constexpr int64_t posInf{std::numeric_limits<int64_t>::max()};
  std::stack<Node*> nodesToExpand;
  spp::sparse_hash_set<uint32_t> visited;
  spp::sparse_hash_map<uint32_t, int64_t> objectiveMap;

  for (const auto& layer : pNodesPerLayer)
  {
    for (auto node : layer)
    {
      objectiveMap[node->getUniqueId()] = posInf;
    }
  }

  // Initialize distances to all vertices as infinite and distance
  // to source as 0
  objectiveMap[pRootNode->getUniqueId()] = 0;
  for (const auto& layer : pNodesPerLayer)
  {
    for (auto node : layer)
    {
      // Update distances of all adjacent vertices
      auto currentNodeCost = objectiveMap[node->getUniqueId()];
      if (currentNodeCost < posInf)
      {
        for (auto edge : node->getOutEdges())
        {
          auto fromState = edge->getTail()->getDPState();
          auto costNextNode = edge->getHead()->getDPState()->cost(edge->getValue(), fromState);
          if (objectiveMap[edge->getHead()->getUniqueId()] > currentNodeCost + costNextNode)
          {
            objectiveMap[edge->getHead()->getUniqueId()] = currentNodeCost + costNextNode;
            edge->getHead()->setSelectedEdge(edge);
          }
        }
      }
    }
  }

  std::vector<Edge*> solution;

  // Start with leaf and trace backwards
  int64_t totCost{0};
  auto node = pNodesPerLayer.at(pNodesPerLayer.size()- 1).at(0);
  while (node->getSelectedEdge() != nullptr)
  {
      totCost += objectiveMap[node->getUniqueId()];
      solution.push_back(node->getSelectedEdge());
      node = node->getSelectedEdge()->getTail();
  }

  std::cout << "Cost: " << totCost << std::endl;
  return solution;
}

void MDD::dfs()
{
  std::stack<Node*> nodesToExpand;
  nodesToExpand.push(pRootNode);
  while(!nodesToExpand.empty())
  {
    auto currNode = nodesToExpand.top();
    nodesToExpand.pop();

    const auto& edges = currNode->getOutEdges();
    if (edges.empty())
    {
      std::cout << "VISIT " << "t" << std::endl;
    }
    else
    {
      std::cout << "VISIT " << currNode->getNodeStringId() << std::endl;
    }
    for (auto it = edges.rbegin(); it != edges.rend(); ++it)
    {
      nodesToExpand.push((*it)->getHead());
    }
  }
}

void MDD::dfsRec(Node* currNode, double& bestCost, const uint32_t maxLayer,
                 double cost, Node* prevNode)
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
    cost += currNode->getDPState()->cost((*it)->getValue(), prevNode->getDPState());
    dfsRec((*it)->getHead(), bestCost, maxLayer, cost, currNode);
  }
}

void MDD::printMDD(const std::string& outFileName)
{
  std::string ppMDD = "digraph D {\n";
  for (int lidx{0}; lidx < static_cast<int>(pNodesPerLayer.size()); ++lidx)
  {
    for (auto node : pNodesPerLayer[lidx])
    {
      for (auto outEdge : node->getOutEdges())
      {
        std::string toNodeId = outEdge->getHead()->getNodeStringId();
        if (lidx == static_cast<int>(pNodesPerLayer.size()) - 2)
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
