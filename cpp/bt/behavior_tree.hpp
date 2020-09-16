//
// Copyright OptiLab 2020. All rights reserved.
//
// Base class for a Behavior Tree.
//

#pragma once

#include <cstdint>  // for uint32_t
#include <limits>   // for std::numeric_limits
#include <memory>   // for std::unique_ptr

#include <sparsepp/spp.h>

#include "bt/behavior_tree_arena.hpp"
#include "bt/blackboard.hpp"
#include "bt/node_status.hpp"
#include "bt/node_status.hpp"
#include "cp/variable.hpp"
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
   BehaviorTree(BehaviorTreeArena::UPtr arena);
   ~BehaviorTree() = default;

   Blackboard::SPtr getBlackboard() const noexcept { return pBlackboard; }

   /// Sets the total number of ticks
   void setTotNumTicks(uint32_t numTicks) noexcept { pTotNumTicks = numTicks; }

   /// Sets the entry node.
   /// The method takes the index of the entry node.
   /// Note: the Behavior Tree doesn't take ownership of the given node
   void setEntryNode(uint32_t entryNode);

   /// Returns the entry node
   uint32_t getEntryNode() const noexcept { return pEntryNode; }

   /// Returns the current status of this BT
   NodeStatus getStatus() const noexcept { return pStatus; }

   /// Returns the raw pointer to the internal arena
   BehaviorTreeArena* getArenaMutable() const noexcept { return pArena.get(); }

   void addVariableMapping(int idx, cp::Variable::SPtr var)
   {
     pVarMap[idx] = var;
   }

   cp::Variable* getVariableMutableGivenOrderingNumber(int idx)
   {
     auto it = pVarMap.find(idx);
     if (it == pVarMap.end())
     {
       return nullptr;
     }
     return (it->second).get();
   }

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

   /// Arena for creating nodes
   BehaviorTreeArena::UPtr pArena{};

   /// Blackboard memory shared among the children of this BT
   Blackboard::SPtr pBlackboard{};

   /// First and unique child of this BT.
   /// All other children are attached to this entry node
   uint32_t pEntryNode{std::numeric_limits<uint32_t>::max()};

   /// Mapping between BT children and model variables.
   /// TODO create a Blackboard specific for optimization and add
   /// this mapping to that blackboard
   spp::sparse_hash_map<int, cp::Variable::SPtr> pVarMap;
};

}  // namespace btsolver

