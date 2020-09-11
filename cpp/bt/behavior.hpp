//
// Copyright OptiLab 2020. All rights reserved.
//
// Behaviors are nodes that contain children.
// Selector and Sequence are standard behavior nodes.
//

#pragma once

#include <cstdio>  // for std::size_t
#include <memory>  // for std::unique_ptr
#include <string>
#include <vector>

#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"

// Forward declarations
namespace btsolver {
class BehaviorTreeArena;
}  // namespace btsolver

namespace btsolver {

class SYS_EXPORT_CLASS Behavior : public Node {
 public:
  using UPtr = std::unique_ptr<Behavior>;
  using SPtr = std::shared_ptr<Behavior>;

 public:
   Behavior(const std::string& name, BehaviorTreeArena* arena, Blackboard* blackboard=nullptr);

   /// Adds a child to this behavior.
   /// Note: children are run in the sequence they are added.
   void addChild(uint32_t childId);

   /// Pops the child from the list of children and returns its unique identifier
   uint32_t popChild();

   /// Replace child "oldChild" with child "newChild"
   void replaceChild(uint32_t oldChild, uint32_t newChild);

   /// Cancel all children currently running
   void cancelChildren();

   /// Cancel run of a particular child
   void cancelChild(uint32_t childId);

   /// Cleanup all active children
   void cleanupChildren();

   /// Resets the current node status for all children
   void resetChildrenStatus();

   /// Forces the current state to CANCEL and calls the cancel
   /// callback, if any
   void cancel() override;

   /// Cleanup performed once run returns a termination value.
   /// This is usually used to reset internal variables
   void cleanup() override;

   /// Returns the list of children of this node
   const std::vector<uint32_t>& getChildren() const noexcept { return pChildren; }

 protected:
   /// Ticks the specified child
   NodeStatus tickChild(uint32_t childId);

   /// Get the child given its identifier
   Node* getChildMutable(uint32_t childId) const;

 private:
   /// List of children nodes
   std::vector<uint32_t> pChildren;

   /// List of open nodes (running nodes)
   std::vector<uint32_t> pOpenNodes;
};

}  // namespace btsolver
