#include "bt_solver/all_different.hpp"

#include <sparsepp/spp.h>

#include "cp/constraint.hpp"

namespace btsolver {
namespace cp {

AllDifferent::AllDifferent(const std::string& name)
: BTConstraint(ConstraintType::kAllDifferent, name)
{
}

btsolver::Sequence* AllDifferent::builBehaviorTreePropagator(BehaviorTreeArena* arena)
{
  const auto& scope = getScope();
  if (arena == nullptr || scope.empty())
  {
    return nullptr;
  }

  auto root = reinterpret_cast<btsolver::Sequence*>(
          arena->buildNode<btsolver::Sequence>("AllDifferent_Root"));

  // The first node is different than other nodes since it contains only
  // states for the domain elements
  auto firstNodeResult = buildFirstNodeBT(scope[0], arena);
  if (firstNodeResult.first == nullptr)
  {
    return nullptr;
  }

  // Add the first node to the root
  root->addChild(firstNodeResult.first->getUniqueId());

  // Work on the other variables in order
  StateMemory loopingStateMemory = firstNodeResult.second;
  for (int idx{1}; idx < static_cast<int>(scope.size()); ++idx)
  {
    auto nodeResult = buildNodeBT(scope[idx], loopingStateMemory, arena);
    if (nodeResult.first == nullptr)
    {
      return nullptr;
    }

    // Attach the selector for the current variable (i.e., "select its value")
    // to the current sequence node
    root->addChild(nodeResult.first->getUniqueId());

    // Update the state memory
    loopingStateMemory = nodeResult.second;
  }

  return root;
}

std::pair<btsolver::Selector*, AllDifferent::StateMemory> AllDifferent::buildFirstNodeBT(
        const Variable::SPtr& var, BehaviorTreeArena* arena)
{
  AllDifferent::StateMemory stateMemory;
  auto selector = reinterpret_cast<btsolver::Selector*>(
          arena->buildNode<btsolver::Selector>("Selector"));

  auto dom = var->getDomainMutable();
  if (dom->isEmpty())
  {
    return {nullptr, stateMemory};
  }

  auto& it = dom->getIterator();
  stateMemory.reserve(dom->size());
  bool breakLoop{false};
  while(true)
  {
    if (it.atEnd())
    {
      // Execute last loop and exit
      breakLoop = true;
    }

    auto stateNode = reinterpret_cast<btsolver::StateNode*>(
            arena->buildNode<btsolver::StateNode>("State"));
    selector->addChild(stateNode->getUniqueId());
    arena->buildEdge(selector, stateNode);

    // Add a connecting edge between the root and the state node.
    // The constructor will automatically register the node on the head and tail nodes
    auto edge = arena->buildEdge(selector, stateNode);

    // Set the domain on this edge
    auto dom = new cp::Domain<cp::BitmapDomain>(it.value());
    edge->setDomainAndOwn(dom);

    // Store the states
    stateMemory.push_back({dom, stateNode});

    it.moveToNext();

    if (breakLoop)
    {
    // Reset iterator and break
      it.reset();
      break;
    }
  }

  return {selector, stateMemory};
}

std::pair<btsolver::Selector*, AllDifferent::StateMemory> AllDifferent::buildNodeBT(
        const Variable::SPtr& var, AllDifferent::StateMemory& stateMemory,
        BehaviorTreeArena* arena)
{
  auto selector = reinterpret_cast<btsolver::Selector*>(
          arena->buildNode<btsolver::Selector>("Selector"));

  auto dom = var->getDomainMutable();
  if (dom->isEmpty())
  {
    return {nullptr, stateMemory};
  }

  // Iterate over all the previous states
  AllDifferent::StateMemory updatedStateMemory;
  for (const auto& it : stateMemory)
  {
    // For each state, create a sequence node
    auto sequence = reinterpret_cast<btsolver::Sequence*>(
            arena->buildNode<btsolver::Sequence>("Sequence_AllDifferent"));

    // Add the sequence to the main selector to return to the root
    selector->addChild(sequence->getUniqueId());

    // Attach a parent node paired with the current state (coming from previous variable)
    auto parentStateNode = reinterpret_cast<btsolver::ParentStateNode*>(
            arena->buildNode<btsolver::ParentStateNode>("ParentState"));
    parentStateNode->pairState(it.second->getUniqueId());
    sequence->addChild(parentStateNode->getUniqueId());

    // Attach a selector node as well
    auto selectorState = reinterpret_cast<btsolver::Selector*>(
            arena->buildNode<btsolver::Selector>("Selector_State"));
    sequence->addChild(selectorState->getUniqueId());

    // Get the difference of the variable domain and the previous state's domain
    auto domDiff = dom->subtract(it.first);

    // For each domain element, add a new state under the selector
    auto& domIt = domDiff->getIterator();
    bool breakLoop{false};
    while(true)
    {
      if (domIt.atEnd())
      {
        breakLoop = true;
      }

      // The new state is {it.first, domIt.value()}.
      // Check if this state is already present (as a permutation) in the new state memory.
      // If so, re-use that state, otherwise create a new one
      auto elements = it.first->getElementList();
      elements.push_back(domIt.value());
      auto stateDomain = new Variable::FiniteDomain(elements);

      btsolver::StateNode* newStateNode{nullptr};
      for (const auto& updatedIt : updatedStateMemory)
      {
        if (updatedIt.first->isEqual(stateDomain))
        {
          newStateNode = updatedIt.second;
          break;
        }
      }

      if (!newStateNode)
      {
        // The state is new
        newStateNode = reinterpret_cast<btsolver::StateNode*>(
                arena->buildNode<btsolver::StateNode>("State"));

        // Add it to the updated memory together with the corresponding domain
        updatedStateMemory.push_back({stateDomain, newStateNode});
      }

      // Add the state node to the selector
      selectorState->addChild(newStateNode->getUniqueId());

      // Add an edge and domain
      auto edge = arena->buildEdge(selector, newStateNode);
      edge->setDomainAndOwn(stateDomain);

      if (breakLoop)
      {
        break;
      }

      domIt.moveToNext();
    }
  }

  return {selector, updatedStateMemory};
}

bool AllDifferent::isFeasible() const noexcept
{
  const auto& scope = getScope();
  spp::sparse_hash_set<uint32_t> valueSet;
  for (const auto& var : scope)
  {
    if (!var->isGround())
    {
      return true;
    }
    const auto val = var->getValue();

    if (valueSet.find(val) != valueSet.end())
    {
      return false;
    }
    else
    {
      valueSet.insert(val);
    }
  }
  return true;
}

}  // namespace cp
}  // namespace btsolver
