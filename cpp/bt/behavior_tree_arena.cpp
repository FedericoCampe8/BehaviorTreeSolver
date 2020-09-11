#include "bt/behavior_tree_arena.hpp"

namespace {
constexpr uint32_t kPreallocatedMemorySize{100};
}  // namespace

namespace btsolver {

BehaviorTreeArena::BehaviorTreeArena()
: pBlackboard(std::make_shared<Blackboard>())
{
  pNodePool.reserve(kPreallocatedMemorySize);
  pEdgePool.reserve(kPreallocatedMemorySize);
}

void BehaviorTreeArena::deleteNode(uint32_t nodeId)
{
  const auto nodeName = pNodePool[pNodeArena.at(nodeId)]->getName();
  pNodePool[pNodeArena.at(nodeId)].reset();
  pNodeArena.erase(nodeId);
}

void BehaviorTreeArena::deleteEdge(uint32_t edgeId)
{
  pEdgePool[pEdgeArena.at(edgeId)].reset();
  pEdgeArena.erase(edgeId);
}

}  // btsolver
