#include "mdd_optimization/all_different.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

// #define DEBUG

namespace mdd {

AllDifferentState::AllDifferentState()
: DPState()
{
}

AllDifferentState::AllDifferentState(const AllDifferentState& other)
{
  pStatesList = other.pStatesList;
}

AllDifferentState::AllDifferentState(AllDifferentState&& other)
{
  pStatesList = std::move(other.pStatesList);
}

AllDifferentState& AllDifferentState::operator=(const AllDifferentState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = other.pStatesList;
  return *this;
}

AllDifferentState& AllDifferentState::operator=(AllDifferentState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = std::move(other.pStatesList);
  return *this;
}

bool AllDifferentState::isEqual(const DPState* other) const noexcept
{
  auto otherDPState = reinterpret_cast<const AllDifferentState*>(other);

  // Check if "other" is contained in this states
  if (pStatesList.size() < otherDPState->pStatesList.size())
  {
    // Return, there is at least one state in "other" that this DP doesn't have
    return false;
  }

  // Check that all states in "other" are contained in this state
  const auto& otherList = otherDPState->pStatesList;
  for (const auto& otherSubList : otherList)
  {
    // Check if this subSet is contained in the other state subset list
    if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
    {
      // State not found
      return false;
    }
  }

  // All states are present
  return true;
}

bool AllDifferentState::isInfeasible() const noexcept
{
  return pStatesList.empty();
}

DPState::SPtr AllDifferentState::next(int64_t domainElement, DPState*) const noexcept
{
  auto state = std::make_shared<AllDifferentState>();
  if (pStatesList.empty())
  {
    state->pStatesList.resize(1);
    state->pStatesList.back().insert(domainElement);
  }
  else
  {
    for (const auto& subSet : pStatesList)
    {
      // Add the new element to all the subset compatible with it
      if (std::find(subSet.begin(), subSet.end(), domainElement) == subSet.end())
      {
        state->pStatesList.push_back(subSet);
        state->pStatesList.back().insert(domainElement);
      }
    }
  }
  return state;
}

double AllDifferentState::cost(int64_t domainElement, DPState*) const noexcept
{
  return static_cast<double>(domainElement);
}

void AllDifferentState::mergeState(DPState* other) noexcept
{
  if (other == nullptr)
  {
    return;
  }

  auto otherDP = reinterpret_cast<const AllDifferentState*>(other);
  for (const auto& otherSubList : otherDP->pStatesList)
  {
    // Check if the other sublist is already present in the current list
    // and, if not, add it
    if (std::find(pStatesList.begin(), pStatesList.end(), otherSubList) == pStatesList.end())
    {
      pStatesList.push_back(otherSubList);
    }
  }
}

std::string AllDifferentState::toString() const noexcept
{
  std::string out{"{"};
  if (pStatesList.empty())
  {
    out += "}";
    return out;
  }

  for (const auto& sublist : pStatesList)
  {
    out += "{";
    for (auto val : sublist)
    {
      out += std::to_string(val) + ", ";
    }
    out.pop_back();
    out.pop_back();
    out += "}, ";
  }
  out.pop_back();
  out.pop_back();
  out += "}";
  return out;
}

AllDifferent::AllDifferent(const std::string& name)
: MDDConstraint(mdd::ConstraintType::kAllDifferent, name),
  pInitialDPState(std::make_shared<AllDifferentState>())
{
}

std::vector<Node*> AllDifferent::mergeNodeSelect(
        int layer,
        const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept
{
  // For the all different, doesn't change much what nodes to select for merging
  std::vector<Node*> nodesToMerge;
  const auto& nodesLayer = mddRepresentation[layer];
  if (nodesLayer.size() < 2)
  {
    return nodesToMerge;
  }
  nodesToMerge.push_back(nodesLayer[0]);
  nodesToMerge.push_back(nodesLayer[1]);

  return nodesToMerge;
}

Node* AllDifferent::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
{
  assert(!nodesList.empty());
  assert(arena != nullptr);

  // For all different, merging nodes selected with the "mergeNodeSelect" means merging
  // DP states on exclusive sets of values (e.g., merging {1, 2} and {1, 3})
  // Pick one at random and set it as the DP state of the new node
  auto mergedNode = arena->buildNode(nodesList.at(0)->getLayer(), nodesList.at(0)->getVariable());
  mergedNode->resetDPState(getInitialDPState());

  for (auto node : nodesList)
  {
    // Merge all nodes DP states
    mergedNode->getDPState()->mergeState(node->getDPState());
  }
  return mergedNode;
}

DPState::SPtr AllDifferent::getInitialDPState() const noexcept
{
  return pInitialDPState;
}


void AllDifferent::enforceConstraint(Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation,
                                     std::vector<Node*>& newNodesList) const
{
    for (int layerIdx{0}; layerIdx < mddRepresentation.size()-1; ++layerIdx) {
        newNodesList.clear();
        for (int nodeIdx{0}; nodeIdx < mddRepresentation.at(layerIdx).size(); ++nodeIdx) {
          auto node = mddRepresentation.at(layerIdx).at(nodeIdx);
          enforceConstraintForNode(node, arena, mddRepresentation, newNodesList);
        }
    }
}


void AllDifferent::enforceConstraintForNode(Node* node, Arena* arena,
                                     std::vector<std::vector<Node*>>& mddRepresentation,
                                     std::vector<Node*>& newNodesList) const
{
  if (node == nullptr)
  {
    throw std::invalid_argument("AllDifferent - enforceConstraint: empty pointer to the node");
  }

  if (arena == nullptr)
  {
    throw std::invalid_argument("AllDifferent - enforceConstraint: empty pointer to the arena");
  }

  // Find all children nodes of the current node
  std::vector<Node*> children;
  children.reserve(node->getOutEdges().size());

  // Use a set for quick lookup to avoid duplicate nodes
  spp::sparse_hash_set<uint32_t> uniqueNodes;
  for (auto edge : node->getOutEdges())
  {
    auto nextNode = edge->getHead();

    // Check that the node hasn't already been added
    if ((uniqueNodes.find(nextNode->getUniqueId()) == uniqueNodes.end()) &&
            (!nextNode->isLeaf()))
    {
      uniqueNodes.insert(nextNode->getUniqueId());
      children.push_back(nextNode);
    }
  }

  // Enforce AllDifferent constraint by splitting nodes
#ifdef DEBUG
  std::cout << "check nodes (num) " << children.size() << std::endl;
#endif

  for (int nodeIdx{0}; nodeIdx < static_cast<int>(children.size()); ++nodeIdx)
  {
    auto nextNode = children.at(nodeIdx);
    const auto& availableValuesTail = node->getValues();
    auto availableValuesHead = nextNode->getValuesMutable();

#ifdef DEBUG
    std::cout << "Check for " << availableValuesTail.size() << " values\n";
#endif

    // Find all conflicting values
    std::vector<int64_t> conflictingValues;
    for (auto val : availableValuesTail)
    {
      if (std::find(availableValuesHead->begin(), availableValuesHead->end(), val) !=
              availableValuesHead->end())
      {

#ifdef DEBUG
        std::cout << "Value is conflicting " << val << std::endl;
#endif
        conflictingValues.push_back(val);
      }
    }

#ifdef DEBUG
    std::cout << "check edges (num) " << node->getOutEdges().size() << std::endl;
#endif

    for (auto edge : node->getOutEdges())
    {
      // If outgoing edge is in conflicting values find its position
      bool foundConflictingEdge{false};
      int64_t conflictingValue;
      for (int idx{0}; idx < static_cast<int>(conflictingValues.size()); ++idx)
      {
        if (conflictingValues.at(idx) == edge->getValue())
        {

#ifdef DEBUG
          std::cout << "Found a conflicting edge " << edge->getValue() << std::endl;
#endif

          conflictingValue = edge->getValue();
          foundConflictingEdge = true;
          break;
        }
      }

      if (foundConflictingEdge)
      {
        // BUG FIX: the domain of newNode was initialized with the full domain of the variable:
        //      newNode->initializeNodeDomain();
        // I believe that it should be initialized with the restricted domain of nextVariable
        auto newReducedDomain = *(node->getValuesMutable());

        // Remove the non admissible value
        auto iterValueNotGood = std::find(newReducedDomain.begin(),
                                          newReducedDomain.end(),
                                          conflictingValue);
        assert(iterValueNotGood != newReducedDomain.end());
        newReducedDomain.erase(iterValueNotGood);

        // BUG FIX: before creating a new node, check if the domain/values that would go into
        // the newNode is already present in another node on the same level.
        // If so, simply connect the edges
        Node* mappedNode{nullptr};
        for (auto newNode : newNodesList)
        {
          // Note: the following works only on ordered list of values.
          // TODO move from lists to sets/bitsets/ranges
          if (newNode->getValues() == newReducedDomain)
          {
            mappedNode = newNode;
            break;
          }
        }

        if (mappedNode != nullptr)
        {
          // A match is found!
          // Connect the edge to the mapped node and continue
          // Move incoming conflicting edge from next node to the mapped node
          nextNode->removeInEdgeGivenPtr(edge);

          // Set the in edge on the new node.
          // Note: the head on the edge is set automatically
          mappedNode->addInEdge(edge);
        }
        else
        {
          // Edge points to conflicting value, so split node
          auto newNode = arena->buildNode(nextNode->getLayer(), nextNode->getVariable());

          // TODO check if it is possible to "build" the domain rather than copying the entire
          // variable domain and removing values
          auto newAvailableValues = newNode->getValuesMutable();
          *newAvailableValues = newReducedDomain;

          // If new node has no available values,
          // then it is infeasible so do not add it to the graph
          if (newAvailableValues->size() > 0)
          {
            // Add the node to the current level
            mddRepresentation[nextNode->getLayer()].push_back(newNode);

            // Store the new node in the newNodesList for next iteration
            newNodesList.push_back(newNode);

            // Move incoming conflicting edge from next node to splitting node
            nextNode->removeInEdgeGivenPtr(edge);

            // Set the in edge on the new node.
            // Note: the head on the edge is set automatically
            newNode->addInEdge(edge);

            // Copy outgoing edges from next node to splitting node
            for (auto outEdge : nextNode->getOutEdges())
            {
              // Add an outgoing edge ONLY IF its correspondent label can be taken from the
              // domain of the curren node
              const auto outValue = outEdge->getValue();
              if (std::find(newAvailableValues->begin(), newAvailableValues->end(), outValue) !=
                      newAvailableValues->end())
              {
                // Build the edge.
                // Note: the constructor will automatically set the pointers to the nodes
                arena->buildEdge(newNode, outEdge->getHead(),
                                 outEdge->getDomainLowerBound(),
                                 outEdge->getDomainUpperBound());
              }
            }
          }
        }

        // If nextNode doesn't have any incoming edge, it can be removed
        // from the layer since it is not reachable anymore
        if (nextNode->getInEdges().empty())
        {
          // Remove all the outgoing edges
          Node::EdgeList outEdges = nextNode->getOutEdges();
          for (auto outEdge : outEdges)
          {
            outEdge->removeEdgeFromNodes();
            arena->deleteEdge(outEdge->getUniqueId());
          }

          // Remove the node from the layer
          auto& nextNodeLayer = mddRepresentation[nextNode->getLayer()];
          auto itNode = std::find(nextNodeLayer.begin(), nextNodeLayer.end(), nextNode);

          assert(itNode != nextNodeLayer.end());
          nextNodeLayer.erase(itNode);
          arena->deleteNode(nextNode->getUniqueId());
        }
      }  // position > -1
    }  // for all out edges
  }  // for all nodes
}

};
