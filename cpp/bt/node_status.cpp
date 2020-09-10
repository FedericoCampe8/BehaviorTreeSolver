#include "bt/node_status.hpp"

namespace btsolver {

NodeStatus::NodeStatus(NodeStatusType status)
: pStatusType(status)
{
}

void NodeStatus::changeStatus(NodeStatusType status) noexcept
{
  pStatusType = status;
}

void NodeStatus::merge(NodeStatusType status) noexcept
{
  if (static_cast<int>(status) > static_cast<int>(pStatusType) &&
          (status != NodeStatusType::kUndefined))
  {
    pStatusType = status;
  }
}

std::string NodeStatus::toString() const noexcept
{
  switch(pStatusType) {
    case NodeStatusType::kActive:
      return "ACTIVE";
    case NodeStatusType::kCancel:
      return "CANCEL";
    case NodeStatusType::kFail:
      return "FAIL";
    case NodeStatusType::kPending:
      return "PENDING";
    case NodeStatusType::kSuccess:
      return "SUCCESS";
    default:
      return "UNDEFINED";
  }
}

}  // namespace btsolver
