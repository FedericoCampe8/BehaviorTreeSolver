#include "bt/node_status.hpp"

namespace btsolver {

std::string statusToString(NodeStatus state) noexcept
{
  switch(state) {
    case NodeStatus::kActive:
      return "ACTIVE";
    case NodeStatus::kCancel:
      return "CANCEL";
    case NodeStatus::kFail:
      return "FAIL";
    case NodeStatus::kPending:
      return "PENDING";
    case NodeStatus::kSuccess:
      return "SUCCESS";
    default:
      return "UNDEFINED";
  }
}

}  // namespace btsolver
