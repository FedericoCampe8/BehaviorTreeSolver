//
// Copyright OptiLab 2020. All rights reserved.
//
// A collection of Behavior Tree nodes implemented for CP solving.
//

#pragma once

#include <atomic>
#include <functional>  // for std::function
#include <iostream>
#include <limits>      // for std::numeric_limits
#include <string>
#include <memory>  // for std::unique_ptr

#include "bt/behavior_tree_arena.hpp"
#include "bt/edge.hpp"
#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "cp/domain.hpp"
#include "system/system_export_defs.hpp"

// Forward declarations
namespace btsolver {
class BehaviorTreeArena;
}  // namespace btsolver

namespace btsolver {

/**
 * \brief A state node is a leaf node that represents a "state"
 *        of the satisfaction/optimization problem w.r.t. the
 *        variables assigned so far and the constraints in the model.
 *        Every time this node runs, it sequentially picks an element
 *        from the domain of the incoming edge and returns SUCCESS.
 *        When there are no more elements, it returns FAIL.
 *
 */
class SYS_EXPORT_CLASS StateNode : public Node {
public:
  using UPtr = std::unique_ptr<StateNode>;

public:
  StateNode(const std::string& name, BehaviorTreeArena* arena, Blackboard* blackboard=nullptr)
  : Node(name, arena, blackboard)
  {
    // Register the config callback
    registerConfigureCallback([=](Blackboard* bb) {
      return this->configNode(bb);
    });

    // Register the run callback
    registerRunCallback([=](Blackboard* bb) {
      return this->runStateNode(bb);
    });
  }

private:
  /// Variable indicating whether a domain is completely explored or not
  std::atomic<bool> pDomainExplored{false};
  cp::Variable::FiniteDomain* pDomain{nullptr};

  void configNode(Blackboard* blackboard)
  {
    // Get the incoming edge
    const auto edgeId = getIncomingEdge();
    if (edgeId == std::numeric_limits<uint32_t>::max())
    {
      return;
    }
    pDomain = getArena()->getEdge(edgeId)->getDomainMutable();
  }

  void cleanupNode(Blackboard* blackboard)
  {
    // Get the incoming edge
    if (pDomain == nullptr)
    {
      return;
    }

    // Reset the domain iterator
    auto& domainIterator  = pDomain->getIterator();
    domainIterator.reset();

    // Reset the domain explored flag
    pDomainExplored = false;
  }

  NodeStatus runStateNode(Blackboard* blackboard)
  {
    if (pDomain == nullptr || pDomainExplored)
    {
      // If a domain is explored, there is nothing left to do: return asap
      return NodeStatus::kFail;
    }

    // Iterate over the domain
    auto& domainIterator  = pDomain->getIterator();

    // Use the current domain value
    std::cout << domainIterator.value() << std::endl;

    if (domainIterator.atEnd())
    {
      // If we are at the end of the domain, set completion done
      pDomainExplored = true;
    }
    else
    {
      // Advance the iterator for next run
      domainIterator.moveToNext();
    }

    // Return success
    return NodeStatus::kSuccess;
  }
};

}  // btsolver
