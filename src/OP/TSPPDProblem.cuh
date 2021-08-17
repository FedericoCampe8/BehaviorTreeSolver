#pragma once

#include <fstream>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <External/Nlohmann/json.hpp>
#include "../DP/Context.h"
#include "Problem.cuh"

namespace OP
{
    class TSPPDProblem: public Problem
    {
        // Members
        public:
        Array<DP::CostType> distances;

        // Functions
        public:
        TSPPDProblem(u32 variablesCount, Memory::MallocType mallocType);
        __host__ __device__ inline DP::CostType getDistance(ValueType from, ValueType to) const;
    };

    template<>
    OP::TSPPDProblem* parseInstance<OP::TSPPDProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::TSPPDProblem::TSPPDProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    distances(variablesCount * variablesCount, mallocType)
{}

__host__ __device__
DP::CostType OP::TSPPDProblem::getDistance(ValueType from, ValueType to) const
{
    return *distances[(from * variables.getCapacity()) + to];
}

OP::TSPPDProblem* OP::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse instance
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OP::TSPPDProblem);
    OP::TSPPDProblem* const problem = reinterpret_cast<OP::TSPPDProblem*>(Memory::safeMalloc(problemSize, mallocType));
    u32 const variablesCount = problemJson["nodes"].size();
    new (problem) OP::TSPPDProblem(variablesCount, mallocType);

    // Init variables
    Variable variable(0,variablesCount - 1);
    for (u32 variableIdx = 0; variableIdx < variablesCount; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Init distances
    for (u32 from = 0; from < variablesCount; from += 1)
    {
        for (u32 to = 0; to < variablesCount; to += 1)
        {
            *problem->distances[(from * variablesCount) + to] = problemJson["edges"][from][to];
        }
    }
    return problem;
}