#include "mdd_optimization/mdd.hpp"

#include <cassert>
#include <stdexcept>  // for std::invalid_argument

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

void MDD::enforceConstraints(Node* relaxedMDD, MDDConstructionAlgorithm algorithmType)
{
  if (relaxedMDD == nullptr)
  {
    throw std::invalid_argument("MDD - enforceConstraints: empty pointer to the MDD");
  }

  if (algorithmType == MDDConstructionAlgorithm::Separation)
  {
    runSeparationProcedure(relaxedMDD);
  }
  else
  {
    runTopDownProcedure(relaxedMDD);
  }
}

Node* MDD::buildRelaxedMDD()
{
  // Build the root node
  pNodesPerLayer.at(kLayerZero).push_back(
          pArena->buildNode(kLayerZero, pProblem->getVariables().at(kLayerZero).get()));

  // Build intermediate layers
  pRootNode = pNodesPerLayer.at(kLayerZero).back();
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
  // tail and head nodes
  auto edge = pArena->buildEdge(node,
                                nextNode,
                                node->getVariable()->getLowerBound(),
                                node->getVariable()->getUpperBound());

  // Return next node
  return nextNode;
}

void MDD::runSeparationProcedure(Node* node)
{

}

void MDD::runTopDownProcedure(Node* node)
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

}  // namespace mdd
