#pragma once

#include <Utils/Algorithms.cuh>
#include "../DD/AuxiliaryData.cuh"
#include "../OP/JSProblem.cuh"
#include "JSPState.cuh"

namespace DP
{
    __host__ __device__ inline DP::CostType calcCost(OP::JSProblem const * problem, JSPState const * currentState, OP::ValueType const value);
    void makeRoot(OP::JSProblem const * problem, JSPState* root);
    __host__ __device__ inline void makeState(OP::JSProblem const * problem, JSPState const * currentState, OP::ValueType value, DP::CostType cost, JSPState* nextState);
    __host__ __device__ inline void mergeState(OP::JSProblem const * problem, JSPState const * currentState, OP::ValueType value, JSPState* nextState);
}

__host__ __device__
DP::CostType DP::calcCost(OP::JSProblem const * problem, JSPState const * currentState, OP::ValueType const value)
{
    OP::ValueType const job = value;
    OP::ValueType const task = job * problem->machines + *currentState->jobs_progress[job];
    u16 const machine = problem->tasks[task]->first;
    u16 const duration = problem->tasks[task]->second;
    DP::CostType const makespan = Algorithms::max(*currentState->machines_makespan[machine], *currentState->jobs_makespan[job]) + duration;
    return Algorithms::max(currentState->cost, makespan);
}


void DP::makeRoot(OP::JSProblem const* problem, JSPState* root)
{
    root->cost = 0;
    for (OP::ValueType job = 0; job < problem->jobs; job += 1)
    {
        *root->jobs_progress[job] = 0;
        *root->jobs_makespan[job] = 0;
        root->admissibleValuesMap.insert(job);
    }
    for (OP::ValueType machine = 0; machine < problem->machines; machine += 1)
    {
        *root->machines_makespan[machine] = 0;
    }
    for(u32 task = 0; task < root->tasks_start.getCapacity(); task +=1 )
    {
        *root->tasks_start[task] = 0;
    }
}

__host__ __device__
void DP::makeState(OP::JSProblem const * problem, JSPState const * currentState, OP::ValueType value, DP::CostType cost, JSPState* nextState)
{
    // Generic
    *nextState = *currentState;
    nextState->cost = cost;
    nextState->admissibleValuesMap.erase(value);
    nextState->selectValue(value);

    // Problem specific
    OP::ValueType const job = value;
    OP::ValueType const task = job * problem->machines + *nextState->jobs_progress[job];
    u16 const machine = problem->tasks[task]->first;
    u16 const duration = problem->tasks[task]->second;

    u16 const task_start = Algorithms::max(*nextState->machines_makespan[machine], *nextState->jobs_makespan[job]);
    *nextState->tasks_start[task] = task_start;

    *nextState->jobs_progress[job] += 1;
    if (*nextState->jobs_progress[job] < problem->machines)
    {
        nextState->admissibleValuesMap.insert(job);
    }

    u16 makespan = task_start + duration;
    *nextState->machines_makespan[machine] = makespan;
    *nextState->jobs_makespan[job] = makespan;

    //nextState->print(true);
}

__host__ __device__
void DP::mergeState(OP::JSProblem const * problem, JSPState const * currentState, OP::ValueType value, JSPState* nextState)
{
    nextState->cost = Algorithms::min(currentState->cost, nextState->cost);
    for (OP::ValueType job = 0; job < problem->jobs; job += 1)
    {
        *nextState->jobs_makespan[job] = Algorithms::min(*currentState->jobs_makespan[job], *nextState->jobs_makespan[job]);
    }
    for (OP::ValueType machine = 0; machine < problem->machines; machine += 1)
    {
        *nextState->machines_makespan[machine] = Algorithms::min(* currentState->machines_makespan[machine], *nextState->machines_makespan[machine]);
    }
}