//
// Copyright OptiLab 2020. All rights reserved.
//
// MDD-based implementation of the
// Job-Shop constraint with availability time windows,
// multiple-machines, and precedence constraints.
//

#pragma once

#include <cstdint>  // for int64_t
#include <memory>
#include <string>
#include <vector>

#include <sparsepp/spp.h>

#include "mdd_optimization/dp_model.hpp"
#include "mdd_optimization/mdd_constraint.hpp"
#include "system/system_export_defs.hpp"

namespace mdd {

/**
 * \brief AllDifferent state used for the DP model
 *        encapsulating the AllDifferent constraint.
 */
class SYS_EXPORT_STRUCT JobShopState : public DPState {
 public:
  /// TaskSpecInfo specifies the task as
  /// [required_machine_id | duration | 10-factor | dependency_task_1 | dependency_task_2 | ...]
  /// Note: duration is duration = original duration * 10-factor (10, 100, 1000, ...)
  using TaskSpecInfo = std::vector<int64_t>;

  /// Task specifications: map from task id to TaskSpecInfo
  using TaskSpecMap = spp::sparse_hash_map<int64_t, TaskSpecInfo>;

  /// Information about a task as <task_id, start_time>
  using TaskInfo = std::pair<int64_t, double>;
  using Schedule = std::vector<TaskInfo>;
  using MachineSchedule = spp::sparse_hash_map<int64_t, Schedule>;

 public:
  JobShopState(TaskSpecMap* taskSpecMap, uint64_t numMachines, bool isDefaultState=false);
  ~JobShopState() = default;

  JobShopState(const JobShopState& other);
  JobShopState(JobShopState&& other);

  JobShopState& operator=(const JobShopState& other);
  JobShopState& operator=(JobShopState&& other);

  void mergeState(DPState* other) noexcept override;

  DPState::SPtr next(int64_t newTaskId, DPState* nextDPState=nullptr) const noexcept override;

  double cost(int64_t newTaskId, DPState* fromState=nullptr) const noexcept override;

  bool isInfeasible() const noexcept override;

  std::string toString() const noexcept override;

  bool isEqual(const DPState* other) const noexcept override;

  bool isMerged() const noexcept override { return false; }

  const MachineSchedule& getSchedule() const noexcept { return pMachineMap; }

 private:
  using MachineCumulativeCost = spp::sparse_hash_map<int64_t, double>;

 private:
  /// Map <machine_id, schedule>.
  /// Note: each machine has its own schedule
  mutable MachineSchedule pMachineMap;

  /// Map <machine_id, cumulative cost>.
  mutable MachineCumulativeCost pMachineCostMap;

  /// Task specification map
  TaskSpecMap* pTaskSpecMap{nullptr};

  /// Scheduled tasks up to this state
  std::vector<int64_t> pGlobalSchedule;
};

class SYS_EXPORT_CLASS JobShop : public MDDConstraint {
 public:
   using UPtr = std::unique_ptr<JobShop>;
   using SPtr = std::shared_ptr<JobShop>;

 public:
   JobShop(const JobShopState::TaskSpecMap& taskSpecMap,
           uint64_t numMachines,
           const std::string& name="JobShop");

   virtual ~JobShop() {}

   /// Enforces this constraint on the given MDD node
   void enforceConstraint(Arena* arena,
                          std::vector<std::vector<Node*>>& mddRepresentation,
                          std::vector<Node*>& newNodesList) const override;

   /// Applies some heuristics to select a subset of nodes in the given layer to merge
   std::vector<Node*> mergeNodeSelect(
           int layer,
           const std::vector<std::vector<Node*>>& mddRepresentation) const noexcept override;

   /// Merges the given list of nodes and returns the representative merged node
   Node* mergeNodes(const std::vector<Node*>& nodesList, Arena* arena) const noexcept override;

   /// Returns the initial DP state
   DPState::SPtr getInitialDPState() const noexcept override;

   /// Returns the initial state of the DP transformation chain as a raw pointer
   DPState* getInitialDPStateRaw() noexcept override { return nullptr; }

   /// Check feasibility of AllDifferent over the variables in its scope
   bool isFeasible() const noexcept override { return true; }

 private:
   /// Task specification map
   JobShopState::TaskSpecMap pTaskSpecMap;

   /// Initial state for the DP model for the JobShop constraint
   JobShopState::SPtr pInitialDPState{nullptr};
};


};
