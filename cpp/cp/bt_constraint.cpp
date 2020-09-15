#include "cp/bt_constraint.hpp"

#include <stdexcept>  // for std::invalid_argument

namespace btsolver {
namespace cp {

BTConstraint::BTConstraint(ConstraintType type, BehaviorTreeArena* arena,
                           const std::string& name)
: Constraint(type, name),
  pArena(arena)
{
  if (pArena == nullptr)
  {
    throw std::invalid_argument("BTConstraint: empty arena");
  }
}

void BTConstraint::buildRelaxedBT()
{
  const auto& scope = getScope();
  if (scope.empty())
  {
    throw std::runtime_error("BTConstraint - buildRelaxedBT: empty constraint scope");
  }

  // Create a sequence node as entry node for the BT
  pSemanticBT = reinterpret_cast<btsolver::Sequence*>(
          pArena->buildNode<btsolver::Sequence>(getName() + "_Root_Sequence"));

  // Add a state domain for each variable with correspondent edge
  for (auto var : getScope())
  {
    auto stateNode = pArena->buildNode<btsolver::StateNode>(getName() + "_State");
    pSemanticBT->addChild(stateNode->getUniqueId());

    // Add a connecting edge between the root and the state node.
    // The constructor will automatically register the node on the head and tail nodes
    auto edge = pArena->buildEdge(pSemanticBT, stateNode);

    // Set the domain on this edge
    edge->setDomain(var->getDomainMutable());
  }
}

}  // namespace cp
}  // namespace btsolver
