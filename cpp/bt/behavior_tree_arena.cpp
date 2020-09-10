#include "bt/behavior_tree_arena.hpp"

namespace {

constexpr uint32_t kPreallocatedMemorySize{100};

}  // namespace

namespace btsolver {

BehaviorTreeArena::BehaviorTreeArena()
{
  pNodePool.reserve(kPreallocatedMemorySize);
  pEdgePool.reserve(kPreallocatedMemorySize);
}

void BehaviorTreeArena::deleteNode(uint32_t nodeId)
{
  const auto nodeName = pNodePool[pNodeArena.at(nodeId)]->getName();
  pNodePool[pNodeArena.at(nodeId)].reset();
  pNodeArena.erase(nodeId);
  pNodeStringArena.erase(nodeName);
}

void BehaviorTreeArena::deleteEdge(uint32_t edgeId)
{
  pEdgePool[pEdgeArena.at(edgeId)].reset();
  pEdgeArena.erase(edgeId);
}

}  // btsolver
