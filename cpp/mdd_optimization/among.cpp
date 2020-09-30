#include "mdd_optimization/among.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <queue>
#include <utility>    // for std::move

// #define DEBUG

namespace {
constexpr int32_t kDefaultBitmapSize{32};
}  // namespace

namespace mdd {

AmongState::AmongState()
: DPState()
{
}

AmongState::AmongState(const AmongState& other)
{
  pStatesList = other.pStatesList;
}

AmongState::AmongState(AmongState&& other)
{
  pStatesList = std::move(other.pStatesList);
}

AmongState& AmongState::operator=(const AmongState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = other.pStatesList;
  return *this;
}

AmongState& AmongState::operator=(AmongState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pStatesList = std::move(other.pStatesList);
  return *this;
}

bool AmongState::isEqual(const DPState* other) const noexcept
{
  auto otherDPState = reinterpret_cast<const AmongState*>(other);

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

bool AmongState::isInfeasible() const noexcept
{
  return pStatesList.empty();
}

DPState::SPtr AmongState::next(int64_t domainElement) const noexcept
{
  auto state = std::make_shared<AmongState>();
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

double AmongState::cost(int64_t domainElement) const noexcept
{
  return static_cast<double>(domainElement);
}

void AmongState::mergeState(DPState* other) noexcept
{
  if (other == nullptr)
  {
    return;
  }

  auto otherDP = reinterpret_cast<const AmongState*>(other);
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

std::string AmongState::toString() const noexcept
{
  std::string out{"{"};
  if (pStatesList.empty())
  {
    out += "}";
    return out;
  }

  for (auto sublist : pStatesList)
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

Among::Among(const std::string& name)
: MDDConstraint(mdd::ConstraintType::kAmong, name),
  pInitialDPState(std::make_shared<AmongState>())
{
}

Among::Among(const std::vector<int64_t>& domain,
             int lower,
             int upper,
             const std::string& name)
: MDDConstraint(mdd::ConstraintType::kAmong, name),
  pInitialDPState(std::make_shared<AmongState>())
{
  setParameters(domain, lower, upper);
}

void Among::setParameters(const std::vector<int64_t>& domain, int lower, int upper)
{
  if (lower > upper)
  {
    throw std::invalid_argument("Among - setParameters: lower bound is greater than upper bound");
  }

  pConstraintDomain = domain;
  pLowerBound = lower;
  pUpperBound = upper;
}

int Among::getConstraintCountForPath(const std::vector<Edge*>& path) const
{
    int count{0};
    for (auto edgeInPath : path)
    {
      // For each edge in the path
      for (auto var : getScope())
      {
        // Check if the variables in the scope of this constraints
        // are at the tail of the current edge
        if (var->getId() == edgeInPath->getTail()->getVariable()->getId())
        {
           count += 1;
           break;
        }
      }
    }

    return count;
}

void Among::enforceConstraintTopDown(Arena* arena,
                                     std::vector<std::vector<Node*>>& mddRepresentation) const
{
  for (int layer{0}; layer < static_cast<int>(mddRepresentation.size()); ++layer)
  {
    bool layerInConstraint{false};
    for (auto var : getScope())
    {
      // Check the first node: all nodes of the same layer share the same variable
      if (mddRepresentation[layer][0]->getVariable()->getId() == var->getId())
      {
        layerInConstraint = true;
        break;
      }
    }

    if (layerInConstraint)
    {
      spp::sparse_hash_map<int, Node*> nodeByConstraintCount;
      for (int nodeIdx{0}; nodeIdx < static_cast<int>(mddRepresentation.at(layer).size()); ++nodeIdx)
      {
        auto node = mddRepresentation.at(layer).at(nodeIdx);
        const auto& nodeInPaths = node->getIncomingPaths();

        // First split and merge nodes according to their paths (state of constraint)
        for (auto inEdge : node->getInEdges())
        {
          // All path to same node should lead to the same count, so use the first one
          const auto& path = nodeInPaths.at(inEdge->getUniqueId()).at(0);

          // Count how many occurrences in a given path.
          // If different paths leads to same number of occurrences,
          // then paths lead to same nodes and should be merged
          int count = getConstraintCountForPath(path);
          if (nodeByConstraintCount.find(count) != nodeByConstraintCount.end() )
          {
            // Equivalent node exist
            inEdge->setHead(nodeByConstraintCount[count]);
          }
          else
          {
            // Create new node for incoming edge
            auto newNode = arena->buildNode(node->getLayer(), node->getVariable());
            newNode->initializeNodeDomain();
            nodeByConstraintCount[count] = newNode;
            inEdge->setHead(newNode);

            // Remove invalid values from new node:
            // for each value in new node, check if that value is contained in node.
            // If so, continue, if not remove it from new node
            const auto& nodeDomain = node->getValues();
            auto newNodeDomain = newNode->getValuesMutable();

            // Keep a copy of the values to remove to avoid invalidating iterators
            std::vector<int64_t> valuesToRemove;
            for (auto val : *(newNodeDomain))
            {
              if (std::find(nodeDomain.cbegin(), nodeDomain.cend(), val) == nodeDomain.cend())
              {
                valuesToRemove.push_back(val);
              }
            }

            for (auto val : valuesToRemove)
            {
              newNodeDomain->erase(std::find(newNodeDomain->begin(), newNodeDomain->end(), val));
            }

            // Copy outgoing edges for new node
            for (auto outEdge : node->getOutEdges())
            {
              arena->buildEdge(newNode,
                               outEdge->getHead(),
                               outEdge->getValue(),
                               outEdge->getValue());
            }
          }  // else an equivalent node does not exists
        }  // for all in edges
      }
    }  // layerInConstraint


    // Splitting and merging nodes for current layer should be consistent at this point
    //---------------------------------------------------------------------------------//

    for (int nodeIdx = 0; nodeIdx < static_cast<int>(mddRepresentation[layer].size()); ++nodeIdx)
    {
      auto node = mddRepresentation.at(layer).at(nodeIdx);
      for (auto outEdge : node->getOutEdges())
      {
        // If edge could create conflict, split into separate node
        if (std::count(pConstraintDomain.begin(), pConstraintDomain.end(), outEdge->getValue()) > 0)
        {
          auto head = outEdge->getHead();
          const auto& headInPaths = head->getIncomingPaths();

          auto inEdge = head->getInEdges().at(0);
          const auto& path = headInPaths.at(inEdge->getUniqueId()).at(0);
          const int count = getConstraintCountForPath(path);

          // If reached upper bound of constraint, this edge is no longer valid
          if (count >= pUpperBound)
          {
            outEdge->removeEdgeFromNodes();
            auto nodeDomain = node->getValuesMutable();
            auto iter = std::find(nodeDomain->begin(), nodeDomain->end(), outEdge->getValue());
            if (iter != nodeDomain->end())
            {
              nodeDomain->erase(iter);
            }

            // The head of the edge could become unreachable if removed only edge leading to it
            // If so, delete it from memory
            if (head->getInEdges().size() == 0)
            {
              arena->deleteNode(head->getUniqueId());
            }

            // Remove the edge from memory
            arena->deleteEdge(outEdge->getUniqueId());
          }
          else
          {
            auto newNode = arena->buildNode(head->getLayer(), head->getVariable());
            newNode->initializeNodeDomain();
            outEdge->setHead(newNode);

            // Remove invalid values from new node:
            // for each value in new node, check if that value is contained in node.
            // If so, continue, if not remove it from new node
            const auto& nodeDomain = head->getValues();
            auto newNodeDomain = newNode->getValuesMutable();

            // Keep a copy of the values to remove to avoid invalidating iterators
            std::vector<int64_t> valuesToRemove;
            for (auto val : *(newNodeDomain))
            {
              if (std::find(nodeDomain.cbegin(), nodeDomain.cend(), val) == nodeDomain.cend())
              {
                valuesToRemove.push_back(val);
              }
            }

            for (auto val : valuesToRemove)
            {
              newNodeDomain->erase(std::find(newNodeDomain->begin(), newNodeDomain->end(), val));
            }

            // Copy outgoing edges for new node
            for (auto outEdge : head->getOutEdges())
            {
              arena->buildEdge(newNode,
                               outEdge->getHead(),
                               outEdge->getValue(),
                               outEdge->getValue());
            }

            // Add new node to the mdd representation
            mddRepresentation[newNode->getLayer()].push_back(newNode);
          }  // else count < pUpperBound

        }
      }  // for all out edges of current node
    }  //  for all nodes in current layer
  }
}

void Among::enforceConstraintBottomUp(
        Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const
{
  int lastNodeLayer =  mddRepresentation.size() - 1;

  // Note: queue is very slow in c++
  std::queue<Node*> queue;

  // Go through each node in the last layer
  for (int nodeIdx = 0; nodeIdx < mddRepresentation.at(lastNodeLayer).size(); ++nodeIdx)
  {
    auto node = mddRepresentation.at(lastNodeLayer).at(nodeIdx);

    spp::sparse_hash_map<int, int> countByValue;
    auto inEdge = node->getInEdges().at(0);

    // Check count in path
    // After top-down all paths to a node should lead to the same count.
    // So it is possible to use any path to count.
    const auto& inPaths = node->getIncomingPaths();
    const auto& path = inPaths.at(inEdge->getUniqueId()).at(0);
    int count = getConstraintCountForPath(path);

    // If count is less than lower bound at last layer, only edges
    // in constraint domain are possible.
    // Assume that there is at least a feasible solution,
    // otherwise leaf node will be disconnected from the rest of the graph
    if (count < pLowerBound)
    {
      for (auto outEdge : node->getOutEdges())
      {
        if (std::count(pConstraintDomain.begin(),
                       pConstraintDomain.end(),
                       outEdge->getValue()) == 0)
        {
          outEdge->removeEdgeFromNodes();
          auto nodeDomain = node->getValuesMutable();
          auto iter = std::find(nodeDomain->begin(), nodeDomain->end(), outEdge->getValue());
          if (iter != nodeDomain->end())
          {
              nodeDomain->erase(iter);
          }

          // Remove the edge
          arena->deleteEdge(outEdge->getUniqueId());
        }
      }
      queue.push(node);
    }
  }

  while(!queue.empty())
  {
    auto curretNode = queue.front();
    queue.pop();

    // Node does not lead to a solution
    if (curretNode->getOutEdges().empty())
    {
      // So delete node and check its parents
      for (auto inEdge : curretNode->getInEdges())
      {
        auto parent = inEdge->getTail();
        queue.push(parent);
        inEdge->removeEdgeFromNodes();

        // Remove the edge
        arena->deleteEdge(inEdge->getUniqueId());
      }
    }
  }

  // At this point only edges leading to feasible solutions should be left
}

void Among::enforceConstraint(Node* node, Arena* arena,
                              std::vector<std::vector<Node*>>& mddRepresentation,
                              std::vector<Node*>& newNodesList) const
{
  if (node == nullptr)
  {
    throw std::invalid_argument("Among - enforceConstraint: empty pointer to the node");
  }

  if (arena == nullptr)
  {
    throw std::invalid_argument("Among - enforceConstraint: empty pointer to the arena");
  }

  // Hackish way of running this only once
  // Current design calls enforce constraint for every node,
  // but this constrain needs a global approach
  // TODO add method inside MddConstraint class returning true if the constraint needs
  // a global approach. Returning false otherwise.
  // Re-write the implementation of the enforce method in the MDD class considering
  // this new constraint query
  if (node->getUniqueId() == mddRepresentation.at(0).at(0)->getUniqueId())
  {
    enforceConstraintTopDown(arena, mddRepresentation);
    enforceConstraintBottomUp(arena, mddRepresentation);
  }
}


std::vector<Node*> Among::mergeNodeSelect(
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

Node* Among::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
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

DPState::SPtr Among::getInitialDPState() const noexcept
{
  return pInitialDPState;
}

};
