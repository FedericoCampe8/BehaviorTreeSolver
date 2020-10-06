#include "mdd_optimization/job_shop.hpp"

#include <algorithm>  // for std::find
#include <cassert>
#include <iostream>
#include <stdexcept>  // for std::invalid_argument
#include <utility>    // for std::move
#include <unordered_set>

// #define DEBUG

namespace {
constexpr int kReqMachineIdx{0};
constexpr int kTaskDurationIdx{1};
constexpr int kTaskFactorIdx{2};
constexpr int kTaskDependencyListIdx{3};
}  // namespace

namespace mdd {

JobShopState::JobShopState(TaskSpecMap* taskSpecMap, uint64_t numMachines, bool isDefaultState)
: DPState(),
  pTaskSpecMap(taskSpecMap)
{
  if (isDefaultState)
  {
    for (uint64_t idx{0}; idx < numMachines; ++idx)
    {
      pMachineCostMap[idx] = 0.0;
    }
  }
}

JobShopState::JobShopState(const JobShopState& other)
{
  pMachineMap = other.pMachineMap;
  pMachineCostMap = other.pMachineCostMap;
  pTaskSpecMap = other.pTaskSpecMap;
  pCost = other.pCost;
  pGlobalSchedule = other.pGlobalSchedule;
}

JobShopState::JobShopState(JobShopState&& other)
{
  pMachineMap = std::move(other.pMachineMap);
  pMachineCostMap = std::move(other.pMachineCostMap);
  pTaskSpecMap = other.pTaskSpecMap;
  pCost = other.pCost;
  pGlobalSchedule = std::move(other.pGlobalSchedule);

  other.pTaskSpecMap = nullptr;
  other.pCost = 0.0;
}

JobShopState& JobShopState::operator=(const JobShopState& other)
{
  if (&other == this)
  {
    return *this;
  }

  pMachineMap = other.pMachineMap;
  pMachineCostMap = other.pMachineCostMap;
  pTaskSpecMap = other.pTaskSpecMap;
  pCost = other.pCost;
  pGlobalSchedule = other.pGlobalSchedule;

  return *this;
}

JobShopState& JobShopState::operator=(JobShopState&& other)
{
  if (&other == this)
  {
    return *this;
  }

  pMachineMap = std::move(other.pMachineMap);
  pMachineCostMap = std::move(other.pMachineCostMap);
  pTaskSpecMap = other.pTaskSpecMap;
  pCost = other.pCost;
  pGlobalSchedule = std::move(other.pGlobalSchedule);

  other.pTaskSpecMap = nullptr;
  other.pCost = 0.0;
  return *this;
}

bool JobShopState::isEqual(const DPState* other) const noexcept
{
  // States are never equal to each other
  return false;
}

bool JobShopState::isInfeasible() const noexcept
{
  // Check if the total makespan is zero
  return pCost == 0.0;
}

DPState::SPtr JobShopState::next(int64_t newTaskId, DPState*) const noexcept
{
  // The given value represents a task to schedule.
  // Get the corresponding specification
  const auto& taskSpec = pTaskSpecMap->at(newTaskId);
  const auto reqMachine = taskSpec[kReqMachineIdx];
  const auto duration = taskSpec[kTaskDurationIdx] / (taskSpec[kTaskFactorIdx] * 1.0);
  auto& newTaskStartTime =  pMachineCostMap[reqMachine];

  auto state = std::make_shared<JobShopState>(
          pTaskSpecMap, static_cast<uint64_t>(pMachineCostMap.size()));

  // Check if:
  // 1) the task has been scheduled already
  auto machineIt = pMachineMap.find(reqMachine);
  if (machineIt != pMachineMap.end())
  {
    // There are already some tasks scheduled for the required machine
    for (const auto& scheduledTask : machineIt->second)
    {
      // For each task info in the schedule, check if the task has been scheduled
      if (scheduledTask.first == newTaskId)
      {
        // If so, return the empty state,
        // since the given task has been already scheduled and
        // cannot be rescheduled again
        return state;
      }
    }
  }

  // Check if:
  // 2) check if the dependencies have been scheduled already
  const auto taskSpecEnd = static_cast<int>(taskSpec.size());
  for (int depIdx{kTaskDependencyListIdx}; depIdx < taskSpecEnd; ++depIdx)
  {
    // Get the schedule for the machine that should run the dependency task
    const auto depTask = taskSpec.at(depIdx);
    const auto depReqMachine = (pTaskSpecMap->at(depTask)).at(kReqMachineIdx);
    const auto depDuration = (pTaskSpecMap->at(depTask)).at(kTaskDurationIdx) /
            ((pTaskSpecMap->at(depTask)).at(kTaskFactorIdx) * 1.0);

    auto machineDepIt = pMachineMap.find(depReqMachine);
    if (machineDepIt == pMachineMap.end())
    {
      // The dependency task has not being scheduled:
      // return the empty -- invalid -- state
      return state;
    }

    // The machine has a schedule, check if the dependency task
    // is part of the schedule and if its completion time finishes before
    // the current task can start.
    // TODO avoid this for-loop and store a map of scheduled tasks
    bool found{false};
    for (const auto& scheduledTask : machineDepIt->second)
    {
      if (scheduledTask.first == depTask)
      {
        // The dependency task is part of the schedule
        if (scheduledTask.second + depDuration > newTaskStartTime)
        {
          // Check if the dependency task finishes before the new task starts.
          // If not, return the empty -- invalid -- state
          return state;
        }
        else
        {
          // The dependency has been correctly scheduled
          found = true;
          break;
        }
      }
    }

    // The dependency has not being found
    if (!found)
    {
      // Return the empty -- invalid -- state
      return state;
    }
  }

  // Here all dependencies have been satisfied and the task can be scheduled
  state->pMachineMap = pMachineMap;
  state->pMachineCostMap = pMachineCostMap;

  // Add the new task to the schedule
  state->pMachineMap[reqMachine].push_back({newTaskId, newTaskStartTime});

  // Increase the makespan on the machine just used
  state->pMachineCostMap[reqMachine] += duration;

  // Update the total makespan
  state->pCost = state->pMachineCostMap[reqMachine] > pCost ?
          state->pMachineCostMap[reqMachine] : pCost;

  // Update the global task schedule
  state->pGlobalSchedule = pGlobalSchedule;
  state->pGlobalSchedule.push_back(newTaskId);

  // Return the new state
  return state;
}

double JobShopState::cost(int64_t newTaskId, DPState*) const noexcept
{
  // Return the cost of scheduling the current task
  const auto& taskSpec = pTaskSpecMap->at(newTaskId);
  const auto reqMachine = taskSpec[kReqMachineIdx];
  const auto duration = taskSpec.at(kTaskDurationIdx) / (taskSpec.at(kTaskFactorIdx) * 1.0);
  return static_cast<double>(pMachineCostMap[reqMachine] + duration);
}

const std::vector<int64_t>& JobShopState::cumulativePath() const noexcept
{
  return pGlobalSchedule;
}

double JobShopState::cumulativeCost() const noexcept
{
  return pCost;
}

void JobShopState::mergeState(DPState*) noexcept
{
  // No merge is possible
  return;
}

std::string JobShopState::toString() const noexcept
{
  std::string out{"{"};
  if (pGlobalSchedule.empty())
  {
    out += "}";
    return out;
  }
  for (auto val : pGlobalSchedule)
  {
    out += std::to_string(val) + ", ";
  }
  out.pop_back();
  out.pop_back();
  out += "}, ";
  return out;
}

JobShop::JobShop(const JobShopState::TaskSpecMap& taskSpecMap,
                 uint64_t numMachines,
                 const std::string& name)
: MDDConstraint(mdd::ConstraintType::kTSPPD, name),
  pTaskSpecMap(taskSpecMap),
  pInitialDPState(std::make_shared<JobShopState>(&pTaskSpecMap, numMachines, true))
{
}

std::vector<Node*> JobShop::mergeNodeSelect(
        int layer,
        const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept
{
  // For the all different, doesn't change much what nodes to select for merging
  std::vector<Node*> nodesToMerge;
  const auto& nodesLayer = mddRepresentation[layer];
  if (nodesLayer.size() < 2)
  {
    return nodesToMerge;
  }
  nodesToMerge.push_back(nodesLayer[0]);
  nodesToMerge.push_back(nodesLayer[1]);

  return nodesToMerge;
}

Node* JobShop::mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept
{
  assert(!nodesList.empty());
  assert(arena != nullptr);

  // For all different, merging nodes selected with the "mergeNodeSelect" means merging
  // DP states on exclusive sets of values (e.g., merging {1, 2} and {1, 3})
  // Pick one at random and set it as the DP state of the new node
  auto mergedNode = arena->buildNode(nodesList.at(0)->getLayer(), nodesList.at(0)->getVariable());
  mergedNode->resetDPState(getInitialDPState());

  for (auto node : nodesList)
  {
    // Merge all nodes DP states
    mergedNode->getDPState()->mergeState(node->getDPState());
  }
  return mergedNode;
}

DPState::SPtr JobShop::getInitialDPState() const noexcept
{
  return pInitialDPState;
}

void JobShop::enforceConstraint(Arena* arena,
                              std::vector<std::vector<Node*>>& mddRepresentation,
                              std::vector<Node*>& newNodesList) const
{
}

};
