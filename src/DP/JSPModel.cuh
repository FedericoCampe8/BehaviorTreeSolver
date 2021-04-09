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
    OP::ValueType const task = value;
    OP::ValueType const job = value / problem->machines;
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
        root->admissibleValuesMap.insert(job * problem->machines); // First task of each job
    }
    for (OP::ValueType machine = 0; machine < problem->machines; machine += 1)
    {
        *root->machines_makespan[machine] = 0;
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
    OP::ValueType const task = value;
    OP::ValueType const job = value / problem->machines;
    u16 const machine = problem->tasks[task]->first;
    u16 const duration = problem->tasks[task]->second;

    *nextState->jobs_progress[job] += 1;
    if (*nextState->jobs_progress[job] < problem->machines)
    {
        nextState->admissibleValuesMap.insert(task + 1);
    }

    u16 const makespan = Algorithms::max(*nextState->machines_makespan[machine], *nextState->jobs_makespan[job]) + duration;
    *nextState->machines_makespan[machine] = makespan;
    *nextState->jobs_makespan[machine] = makespan;
    //nextState->print(true);
}

__host__ __device__
void DP::mergeState(OP::JSProblem const * problem, JSPState const * currentState, OP::ValueType value, JSPState* nextState)
{
   // Not implemented
}