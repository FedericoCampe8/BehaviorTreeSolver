#pragma once

#include <fstream>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <External/Nlohmann/json.hpp>
#include "../DP/Context.h"
#include "Problem.cuh"

namespace OP
{
    class SOPProblem: public Problem
    {
        // Members
        public:
        Array<DP::CostType> distances;

        // Functions
        public:
        SOPProblem(u32 variablesCount, Memory::MallocType mallocType);
        __host__ __device__ inline DP::CostType getDistance(ValueType from, ValueType to) const;
    };

    template<>
    OP::SOPProblem* parseInstance<OP::SOPProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::SOPProblem::SOPProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    distances(variablesCount * variablesCount, mallocType)
{}

__host__ __device__
DP::CostType OP::SOPProblem::getDistance(ValueType from, ValueType to) const
{
    return *distances[(from * variables.getCapacity()) + to];
}

OP::SOPProblem* OP::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse instance
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OP::SOPProblem);
    OP::SOPProblem* const problem = reinterpret_cast<OP::SOPProblem*>(Memory::safeMalloc(problemSize, mallocType));
    u32 const nodes = problemJson["nodes"];
    new (problem) OP::SOPProblem(nodes, mallocType);

    // Init variables
    Variable variable(0,nodes-1);
    for (u32 variableIdx = 0; variableIdx < nodes; variableIdx += 1)
    {
        problem->add(&variable);
    }
    //problem->variables[0]->maxValue = 0;
    //problem->variables[nodes-1]->minValue = nodes - 1;

    // Init distances
    for (u32 from = 0; from < nodes; from += 1)
    {
        for (u32 to = 0; to < nodes; to += 1)
        {
            *problem->distances[(from * nodes) + to] = problemJson["edges"][from][to];
        }
    }
    return problem;
}