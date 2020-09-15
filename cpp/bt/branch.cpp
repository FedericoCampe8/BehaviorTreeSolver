#include "bt/branch.hpp"

#include <functional>  // for std::function
#include <iostream>

namespace btsolver {

Selector::Selector(const std::string& name, BehaviorTreeArena* arena, Blackboard* blackboard)
: Behavior(name, arena, blackboard)
{
  // Register the run callback
  registerRunCallback([=](Blackboard* bb) {
    return this->runSelector(bb);
  });
}

NodeStatus Selector::runSelector(Blackboard* blackboard)
{
  // Reset the status of all children before running
  resetChildrenStatus();

  // Run one child at a time, from left to right
  for (auto& child : getChildren())
  {
    auto result = tickChild(child);
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

Sequence::Sequence(const std::string& name, BehaviorTreeArena* arena, Blackboard* blackboard)
: Behavior(name, arena, blackboard)
{
  // Register the run callback
  registerRunCallback([=](Blackboard* bb) {
    return this->runSequence(bb);
  });
}

NodeStatus Sequence::runSequence(Blackboard* blackboard)
{
  // Reset the status of all children before running
  resetChildrenStatus();

  // Run one child at a time, from left to right
  for (auto& child : getChildren())
  {
    const auto result = tickChild(child);
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
