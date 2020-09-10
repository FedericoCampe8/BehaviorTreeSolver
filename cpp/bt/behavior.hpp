//
// Copyright OptiLab 2020. All rights reserved.
//
// Behaviors are nodes that contain children.
// Selector and Sequence are standard behavior nodes.
//

#pragma once

#include <cstdio>  // for std::size_t
#include <string>
#include <vector>

#include <sparsepp/spp.h>

#include "bt/node.hpp"


namespace btsolver {

class SYS_EXPORT_CLASS Behavior : public Node {
 public:
  using UPtr = std::unique_ptr<Behavior>;
  using SPtr = std::shared_ptr<Behavior>;

 public:
   Behavior(const std::string& name);

   /// Adds a child to this behavior.
   /// Notice: children are run in the sequence they are added
   void addChild(Node::UPtr child);

   /// Pops the child from the list of children and returns its reference
   Node::UPtr popChild();

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

 protected:
   /// Ticks the specified child
   void tickChild(uint32_t childId);

   /// Get the child given its identifier
   Node* getChildMutable(uint32_t childId) const;

 private:
   using ChildrenNodeMap = spp::sparse_hash_map<uint32_t, std::size_t>;

 private:
   /// List of children nodes
   std::vector<Node::UPtr> pChildren;

   /// Map of node's unique identifiers to the pointers in the vector
   ChildrenNodeMap pChildrenMap;

   /// List of open nodes (running nodes)
   std::vector<uint32_t> pOpenNodes;
};

}  // namespace btsolver
