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
    class OSSProblem : public Problem
    {
        // Members
        public:
        OP::ValueType const jobs;
        OP::ValueType const machines;
        Array<u16> tasks;

        // Functions
        public:
        OSSProblem(u32 jobs, u32 machines, Memory::MallocType mallocType);
    };

    template<>
    OP::OSSProblem* parseInstance<OSSProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::OSSProblem::OSSProblem(u32 jobs, u32 machines, Memory::MallocType mallocType) :
    Problem(jobs * machines, mallocType),
    jobs(jobs),
    machines(machines),
    tasks(jobs * machines, mallocType)
{}

OP::OSSProblem* OP::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse json
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OP::OSSProblem);
    OP::OSSProblem* const problem = reinterpret_cast<OP::OSSProblem*>(Memory::safeMalloc(problemSize, mallocType));
    OP::ValueType const jobs = problemJson["jobs"];
    OP::ValueType const machines = problemJson["machines"];
    new (problem) OP::OSSProblem(jobs, machines, mallocType);

    // Init variables
    Variable const variable(0, jobs * machines - 1);
    for (OP::ValueType variableIdx = 0; variableIdx < jobs * machines; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Init tasks
    for (OP::ValueType job = 0; job < jobs; job += 1)
    {
        for (OP::ValueType machine = 0; machine < machines; machine += 1)
        {
            u16 const task = (job * machines) + machine;
            *problem->tasks[task] = problemJson["tasks"][job][machine]; // Duration
        }
    }
    return problem;
}