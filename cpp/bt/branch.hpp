//
// Copyright OptiLab 2020. All rights reserved.
//
// Collection of branch nodes.
//

#pragma once

#include <memory>  // for std::unique_ptr
#include <string>

#include "bt/behavior.hpp"
#include "bt/node_status.hpp"

namespace btsolver {

/**
 * \brief A Selector runs each child in order until one succeeds.
 *        This allows failover behaviors.  If one succeeds,
 *        execution of this behavior stops and it returns NodeStatus.SUCCESS.
 *        If no child succeed, Selector returns NodeStatus.FAIL.
 *        If a child is still ACTIVE, then a NodeStatus.ACTIVE status is returned.
 */
class SYS_EXPORT_CLASS Selector : public Behavior {
 public:
   using UPtr = std::unique_ptr<Selector>;
   using SPtr = std::shared_ptr<Selector>;

 public:
   Selector(const std::string& name);

 private:
   NodeStatus runSelector(const Blackboard::SPtr& blackboard);
};

/**
 * \brief A Sequence runs through each child unless one fails.
 *        If one fails, the executiong of this behavior stops and it returns NodeStatus.FAIL.
 *        This behavior returns NodeStatus.ACTIVE until all children have finished
 *        or one failed.  It returns NodeStatus.SUCCESS if all children succeed.
 */
class SYS_EXPORT_CLASS Sequence : public Behavior {
 public:
   using UPtr = std::unique_ptr<Sequence>;
   using SPtr = std::shared_ptr<Sequence>;

 public:
   Sequence(const std::string& name);

 private:
   NodeStatus runSequence(const Blackboard::SPtr& blackboard);
};

}  // namespace btsolver
