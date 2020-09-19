#include "bt/behavior.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <limits>     // for std::numeric_limits
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

#include "bt/behavior_tree_arena.hpp"
#include "bt/edge.hpp"

namespace btsolver {

Behavior::Behavior(const std::string& name, NodeType nodeType, BehaviorTreeArena* arena)
: Node(name, nodeType, arena)
{
}

Edge* Behavior::addChild(Node* child)
{
  // Insert the children in the list of children
  if (child != nullptr)
  {
    // Create an edge connecting the two nodes
    return getArena()->buildEdge(this, child);
  }
  return nullptr;
}

std::pair<Node*, Edge*> Behavior::popChild() noexcept
{
  if (pOutgoingEdges.empty())
  {
    return {nullptr, nullptr};
  }

  // Pop the node from the list of children
  // by removing the connecting edge
  auto edge = pOutgoingEdges.back();
  removeOutgoingEdge(edge);

  auto outNode = edge->getTail();
  edge->removeTail();

  // Return the node (identifier)
  return {outNode, edge};
}

Behavior::ChildrenList Behavior::getChildren() const noexcept
{
  ChildrenList list;
  list.reserve(pOutgoingEdges.size());
  for (auto edge : pOutgoingEdges)
  {
    assert(edge->getTail() != nullptr);
    list.push_back(edge->getTail());
  }
  return list;
}

NodeStatus Behavior::tickChild(Node* child)
{
  assert(child != nullptr);

  // Run the node and get its result
  const auto result = child->tick();
  if (result == NodeStatus::kActive)
  {
    if (std::find(pOpenNodes.begin(), pOpenNodes.end(), child) == pOpenNodes.end())
    {
      // The node is still active, add it to the set of open nodes (if not already present)
      pOpenNodes.push_back(child);
    }
  }
  else if (result == NodeStatus::kPending)
  {
    auto iter = std::find(pOpenNodes.begin(), pOpenNodes.end(), child);
    if (iter != pOpenNodes.end())
    {
      // The node is can be removed since is not longer opened
      pOpenNodes.erase(iter);
    }
  }
  return result;
}

void Behavior::cancelChildren()
{
  for (auto edge : pOutgoingEdges)
  {
    assert(edge->getTail() != nullptr);
    this->cancelChild(edge->getTail());
  }
}

void Behavior::cancel()
{
  this->cancelChildren();
  Node::cancel();
}

void Behavior::cancelChild(Node* child)
{
  if (child == nullptr)
  {
    return;
  }

  if (child->getResult() != NodeStatus::kPending)
  {
    child->cancel();
  }
}

void Behavior::cleanupChildren()
{
  for (auto edge : pOutgoingEdges)
  {
    assert(edge->getTail() != nullptr);
    if (edge->getTail()->getResult() != NodeStatus::kPending)
    {
      edge->getTail()->cleanup();
      auto iter = std::find(pOpenNodes.begin(), pOpenNodes.end(), edge->getTail());
      if (iter != pOpenNodes.end())
      {
        // The node is can be removed since is not longer opened
        pOpenNodes.erase(iter);
      }
    }
  }
}

void Behavior::cleanup()
{
  this->cancelChildren();
  this->cleanupChildren();
  Node::cleanup();
}

}  // namespace btsolver
