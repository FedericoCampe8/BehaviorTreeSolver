#include "bt/branch.hpp"

#include <cassert>
#include <functional>  // for std::function
#include <iostream>

#include "bt/behavior_tree_arena.hpp"

namespace btsolver {

Selector::Selector(const std::string& name, BehaviorTreeArena* arena)
: Behavior(name, NodeType::Selector, arena)
{
  // Register the run callback
  registerRunCallback([=]() {
    return this->runSelector();
  });
}

NodeStatus Selector::runSelector()
{
  // Reset the status of all children before running
  resetChildrenStatus();

  // Run one child at a time, from left to right
  for (auto& edge : pOutgoingEdges)
  {
    assert(edge->getTail());
    auto result = tickChild(edge->getTail());
    if (result == NodeStatus::kActive || result == NodeStatus::kPending)
    {
      // The current child is still active, return asap
      return NodeStatus::kActive;
    }

    if (result == NodeStatus::kSuccess)
    {
      // One child (the first found from the left) succeeded, return success
      // without executing any other children
      return NodeStatus::kSuccess;
    }
  }

  // All children fail
  return NodeStatus::kFail;
}

Sequence::Sequence(const std::string& name, BehaviorTreeArena* arena)
: Behavior(name, NodeType::Sequence, arena)
{
  // Register the run callback
  registerRunCallback([=]() {
    return this->runSequence();
  });
}

NodeStatus Sequence::runSequence()
{
  // Reset the status of all children before running
  resetChildrenStatus();

  // Run one child at a time, from left to right
  for (auto& edge : pOutgoingEdges)
  {
    assert(edge->getTail());
    auto result = tickChild(edge->getTail());
    if (result == NodeStatus::kActive || result == NodeStatus::kPending)
    {
      // The current child is still active, return asap
      return NodeStatus::kActive;
    }

    if (result != NodeStatus::kSuccess)
    {
      // Return on the first node that failed (from the left)
      return NodeStatus::kFail;
    }
  }

  // All children succeeded
  return NodeStatus::kSuccess;
}

}  // namespace btsolver
