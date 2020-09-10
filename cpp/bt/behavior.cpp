#include "bt/behavior.hpp"

#include <algorithm>  // for std::find
#include <limits>     // for std::numeric_limits
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

#include "bt/behavior_tree_arena.hpp"

namespace {
constexpr std::size_t kDefaultNumChildren{16};
}  // namespace

namespace btsolver {

Behavior::Behavior(const std::string& name, BehaviorTreeArena* arena)
: Node(name, arena)
{
  pChildren.reserve(kDefaultNumChildren);
  pOpenNodes.reserve(kDefaultNumChildren);
}

void Behavior::addChild(uint32_t childId)
{
  // Insert the children in the list of children
  pChildren.push_back(childId);
}

uint32_t Behavior::popChild()
{
  if (pChildren.empty())
  {
    return std::numeric_limits<uint32_t>::max();
  }

  // Pop the node from the list of children
  auto outNode = pChildren.back();
  pChildren.pop_back();

  // Return the node (identifier)
  return outNode;
}

Node* Behavior::getChildMutable(uint32_t childId) const
{
  return getArena()->getNode(childId);
}

NodeStatus Behavior::tickChild(uint32_t childId)
{
  auto child = this->getChildMutable(childId);

  // Run the node and get its result
  const auto result = child->tick();
  if (result == NodeStatus::kActive)
  {
    auto nodeId = child->getUniqueId();
    if (std::find(pOpenNodes.begin(), pOpenNodes.end(), nodeId) == pOpenNodes.end())
    {
      // The node is still active, add it to the set of open nodes (if not already present)
      pOpenNodes.push_back(nodeId);
    }
  }
  else if (result == NodeStatus::kPending)
  {
    auto nodeId = child->getUniqueId();
    auto iter = std::find(pOpenNodes.begin(), pOpenNodes.end(), nodeId);
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
  for (auto& child : pChildren)
  {
    this->cancelChild(child);
  }
}

void Behavior::cancel()
{
  this->cancelChildren();
  Node::cancel();
}

void Behavior::cancelChild(uint32_t childId)
{
  auto child = this->getChildMutable(childId);
  if (child->getResult() != NodeStatus::kPending)
  {
    child->cancel();
  }
}

void Behavior::cleanupChildren()
{
  for (auto& childId : pChildren)
  {
    auto child = getChildMutable(childId);
    if (child->getResult() != NodeStatus::kPending)
    {
      child->cleanup();
      auto nodeId = child->getUniqueId();
      auto iter = std::find(pOpenNodes.begin(), pOpenNodes.end(), nodeId);
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

void Behavior::resetChildrenStatus()
{
  for (auto& child : pChildren)
  {
    // TODO with blackboard
  }
}

}  // namespace btsolver
