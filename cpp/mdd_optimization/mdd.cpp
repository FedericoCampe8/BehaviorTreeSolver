#include "mdd_optimization/mdd.hpp"

#include <sparsepp/spp.h>

#include <cassert>
#include <iostream>
#include <stack>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::pair

// #define DEBUG

namespace {
constexpr uint32_t kLayerZero{0};
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
  else
  {
    // Start from the root node
    auto rootNode = buildRootMDD();
    runTopDownProcedure(rootNode);
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
  }
}

void MDD::runSeparationProcedureOnConstraint(Node* root, MDDConstraint* con)
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

  // Step 3: for each layer, for each node, for each arc,
  //         compute the next DP state and split nodes accordingly
  auto totLayers = static_cast<uint32_t>(pProblem->getVariables().size());
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
        // by applying the given edge/arc value
        auto newDPState = currDPState->next(edgeValue);
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

void MDD::runTopDownProcedure(Node* node)
{
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

  // For all layers
  std::vector<std::pair<DPState*, Node*>> newDPStates;
  newDPStates.reserve(pMaxWidth);
  const auto totLayers = static_cast<int>(pProblem->getVariables().size());
  for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
  {
    // For all nodes per layer
    newDPStates.clear();
    for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
    {
      // For all values of the domain of the current layer
      auto currNode = pNodesPerLayer.at(layerIdx).at(nodeIdx);
      auto currDomain = currNode->getValues();
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
}

void MDD::runFilteringProcedure(Node* node)
{
  // Enforce all constraints
  auto totLayers = static_cast<uint32_t>(pProblem->getVariables().size());
  for (auto& con : pProblem->getConstraints())
  {
    for (int layerIdx{0}; layerIdx < totLayers; ++layerIdx)
    {
      for (int nodeIdx{0}; nodeIdx < pNodesPerLayer.at(layerIdx).size(); ++nodeIdx)
      {
        auto node = pNodesPerLayer.at(layerIdx).at(nodeIdx);
        con->enforceConstraint(node);
      }
    }
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
                currentNode->getDPState()->cost(edge->getValue()));
        nextNode->setSelectedEdge(edge);
      }
      else
      {
        // If it does, the node has been visited already,
        // update its partial solution choosing the maximum value
        double candidateValue = currentNode->getOptimizationValue() +
                currentNode->getDPState()->cost(edge->getValue());
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

}  // namespace mdd