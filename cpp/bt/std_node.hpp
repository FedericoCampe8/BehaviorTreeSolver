//
// Copyright OptiLab 2020. All rights reserved.
//
// A collection of standard Behavior Tree nodes.
//

#pragma once

#include <functional>  // for std::function
#include <iostream>
#include <string>
#include <memory>  // for std::unique_ptr

#include "bt/node.hpp"
#include "bt/node_status.hpp"
#include "system/system_export_defs.hpp"

// Forward declarations
namespace btsolver {
class BehaviorTreeArena;
}  // namespace btsolver

namespace btsolver {

/**
 * \brief A node that logs text on standard output and returns SUCCESS.
 */
class SYS_EXPORT_CLASS LogNode : public Node {
public:
  using UPtr = std::unique_ptr<LogNode>;

public:
  LogNode(const std::string& name, BehaviorTreeArena* arena)
  : Node(name, arena)
  {
    // Register the run callback
    std::function<NodeStatus(const Blackboard::SPtr&)> callback;
    registerRunCallback([=](const Blackboard::SPtr& bb) {
      return this->runLog(bb);
    });
  }

  void setLog(const std::string& log) noexcept { pLog = log; }

private:
  std::string pLog{};
  NodeStatus runLog(const Blackboard::SPtr& blackboard)
  {
    std::cout << pLog << std::endl;
    return NodeStatus::kSuccess;
  }
};

}  // btsolver
