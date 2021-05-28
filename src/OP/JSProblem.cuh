#pragma once

#include <fstream>
#include <Containers/Array.cuh>
#include <Containers/Pair.cuh>
#include <Containers/Triple.cuh>
#include <External/Nlohmann/json.hpp>
#include <Utils/Algorithms.cuh>
#include "../DP/Context.h"
#include "Problem.cuh"

namespace OP
{
    class JSProblem : public Problem
    {
        // Members
        public:
        OP::ValueType const jobs;
        OP::ValueType const machines;
        Array<Pair<u16>> tasks;

        // Functions
        public:
        JSProblem(u32 jobs, u32 machines, Memory::MallocType mallocType);
    };

    template<>
    OP::JSProblem* parseInstance<JSProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::JSProblem::JSProblem(u32 jobs, u32 machines, Memory::MallocType mallocType) :
    Problem(jobs * machines, mallocType),
    jobs(jobs),
    machines(machines),
    tasks(jobs * machines, mallocType)
{}

OP::JSProblem* OP::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse json
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OP::JSProblem);
    OP::JSProblem* const problem = reinterpret_cast<OP::JSProblem*>(Memory::safeMalloc(problemSize, mallocType));
    OP::ValueType const jobs = problemJson["jobs"];
    OP::ValueType const machines = problemJson["machines"];
    new (problem) OP::JSProblem(jobs, machines, mallocType);

    // Init variables
    Variable const variable(0, jobs - 1);
    for (OP::ValueType variableIdx = 0; variableIdx < jobs * machines; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Init orders
    for (OP::ValueType job = 0; job < jobs; job += 1)
    {
        for (OP::ValueType machine = 0; machine < machines; machine += 1)
        {
            u16 const taskIdx = (job * machines) + machine;
            problem->tasks[taskIdx]->first = problemJson["tasks"][job][machine][0]; // Machine
            problem->tasks[taskIdx]->second = problemJson["tasks"][job][machine][1]; // Duration
        }
    }
    return problem;
}