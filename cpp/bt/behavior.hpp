//
// Copyright OptiLab 2020. All rights reserved.
//
// Behaviors are nodes that contain children.
// Selector and Sequence are standard behavior nodes.
//

#pragma once

#include <cstdio>   // for std::size_t
#include <memory>   // for std::unique_ptr
#include <string>
#include <utility>  // for std::pair
#include <vector>

#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"

// Forward declarations
namespace btsolver {
class BehaviorTreeArena;
class Edge;
}  // namespace btsolver

namespace btsolver {

class SYS_EXPORT_CLASS Behavior : public Node {
 public:
  using ChildrenList = std::vector<Node*>;
  using UPtr = std::unique_ptr<Behavior>;
  using SPtr = std::shared_ptr<Behavior>;

 public:
   Behavior(const std::string& name, NodeType nodeType, BehaviorTreeArena* arena);

   /// Adds a child to this behavior creating an edge between the two nodes.
   /// Returns the edge created to link the two nodes.
   /// @note children are run in the sequence they are added.
   Edge* addChild(Node* child);

   /// Pops the child from the list of children,
   /// removes the edge between the two nodes,
   /// and returns the removed child, and the connecting edge
   std::pair<Node*, Edge*> popChild() noexcept;

   /// Cancel all children currently running
   void cancelChildren();

   /// Cancel run of a particular child
   void cancelChild(Node* child);

   /// Cleanup all active children
   void cleanupChildren();

   /// Resets the current node status for all children
   void resetChildrenStatus() {}

   /// Forces the current state to CANCEL and calls the cancel
   /// callback, if any
   void cancel() override;

   /// Cleanup performed once run returns a termination value.
   /// This is usually used to reset internal variables
   void cleanup() override;

   /// Returns the list of children of this node
   ChildrenList getChildren() const noexcept;

 protected:
   /// Ticks the specified child
   NodeStatus tickChild(Node* child);

 private:
   /// List of open nodes (running nodes)
   ChildrenList pOpenNodes;
};

}  // namespace btsolver
