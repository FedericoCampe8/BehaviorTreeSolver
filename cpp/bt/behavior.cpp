#include "bt/behavior.hpp"

#include <algorithm>  // for std::find
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move

namespace {
constexpr std::size_t kDefaultNumChildren{16};
}  // namespace

namespace btsolver {

Behavior::Behavior(const std::string& name)
: Node(name)
{
  pChildren.reserve(kDefaultNumChildren);
  pOpenNodes.reserve(kDefaultNumChildren);
}

void Behavior::addChild(Node::UPtr child)
{
  if (child == nullptr)
  {
    throw std::invalid_argument("Behavior - addChild: empty child instance");
  }

  // Fail if element is already present in the map
  pChildrenMap.insert(std::make_pair(child->getUniqueId(), pChildren.size()));

  // Insert the children in the map
  pChildren.push_back(std::move(child));
}

Node::UPtr Behavior::popChild()
{
  if (pChildrenMap.empty())
  {
    return nullptr;
  }

  // Pop the node from the list of children
  auto outNode = std::move(pChildren.back());
  pChildren.pop_back();

  // Remove the entry from the map
  pChildrenMap.erase(outNode->getUniqueId());

  // Return the node (pointer)
  return std::move(outNode);
}

Node* Behavior::getChildMutable(uint32_t childId) const
{
  auto childPos = pChildrenMap.find(childId);
  if (childPos == pChildrenMap.end())
  {
    throw std::runtime_error("Behavior - tickChild: child not found");
  }

  return pChildren.at(childPos->second).get();
}

void Behavior::tickChild(uint32_t childId)
{
  auto child = this->getChildMutable(childId);

  // Run the node
  child->tick();

  // Get the result status of the node
  const auto result = (child->getResult()).getStatus();
  if (result == NodeStatus::NodeStatusType::kActive)
  {
    auto nodeId = child->getUniqueId();
    if (std::find(pOpenNodes.begin(), pOpenNodes.end(), nodeId) == pOpenNodes.end())
    {
      // The node is still active, add it to the set of open nodes
      pOpenNodes.push_back(nodeId);
    }
  }
  else if (result == NodeStatus::NodeStatusType::kPending)
  {
    auto nodeId = child->getUniqueId();
    auto iter = std::find(pOpenNodes.begin(), pOpenNodes.end(), nodeId);
    if (iter != pOpenNodes.end())
    {
      // The node is can be removed since is not longer opened
      pOpenNodes.erase(iter);
    }
  }
}

void Behavior::cancelChildren()
{
  for (auto& child : pChildren)
  {
    this->cancelChild(child->getUniqueId());
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
  if (child->getResult().getStatus() != NodeStatus::NodeStatusType::kPending)
  {
    child->cancel();
  }
}

void Behavior::cleanupChildren()
{
  for (auto& child : pChildren)
  {
    if (child->getResult().getStatus() != NodeStatus::NodeStatusType::kPending)
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
