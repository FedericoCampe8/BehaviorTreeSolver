//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Behavior Tree.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <memory>   // for std::unique_ptr

#include "bt/blackboard.hpp"
#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"

namespace btsolver {

/**
 * \brief Main class for Behavior Trees.
 * A Behavior Tree (BT) always contains a root node.
 * The BT ticks with a certain frequency or with system frequency.
 */
class SYS_EXPORT_CLASS BehaviorTree {
 public:
   using UPtr = std::unique_ptr<BehaviorTree>;
   using SPtr = std::shared_ptr<BehaviorTree>;

 public:
   BehaviorTree() = default;
   ~BehaviorTree() = default;

   void setBlackboard(Blackboard::SPtr blackboard) noexcept { pBlackboard = blackboard; }
   Blackboard::SPtr getBlackboard() const noexcept { return pBlackboard; }

   /// Sets the total number of ticks
   void setTotNumTicks(uint32_t numTicks) noexcept { pTotNumTicks = numTicks; }

   /// Sets the entry node
   void setEntryNode(Node::UPtr entryNode);

   /// Returns the current status of this BT
   NodeStatus getStatus() const noexcept { return pStatus; }

   /// Runs this behavior tree
   void run();

   /// Stops execution of this BT
   void stop() { pStopRun = true; }

 private:
   /// Total number of ticks to run this BT on.
   /// If this number is zero, the BT runs until
   /// the entry node status is not active
   uint32_t pTotNumTicks{0};

   /// Flag indicating whether or not BT execution should stop
   bool pStopRun{false};

   /// Status of the BT w.r.t. its children
   NodeStatus pStatus{NodeStatus::kActive};

   /// Blackboard memory shared among the children of this BT
   Blackboard::SPtr pBlackboard{};

   /// First and unique child of this BT.
   /// All other children are attached to this entry node
   Node::UPtr pEntryNode{};
};

}  // namespace btsolver

