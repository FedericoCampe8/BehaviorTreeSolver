#pragma once

#include <fstream>
#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <External/Nlohmann/json.hpp>
#include "../DP/Context.h"
#include "Problem.cuh"

namespace OP
{
    class VRProblem: public Problem
    {
        // Members
        public:
        Array<DP::CostType> distances;

        // Functions
        public:
        VRProblem(u32 variablesCount, Memory::MallocType mallocType);
        __host__ __device__ inline DP::CostType getDistance(ValueType from, ValueType to) const;
    };

    template<>
    OP::VRProblem* parseInstance<OP::VRProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::VRProblem::VRProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    distances(variablesCount * variablesCount, mallocType)
{}

__host__ __device__
DP::CostType OP::VRProblem::getDistance(ValueType from, ValueType to) const
{
    return *distances[(from * variables.getCapacity()) + to];
}

OP::VRProblem* OP::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse instance
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OP::VRProblem);
    OP::VRProblem* const problem = reinterpret_cast<OP::VRProblem*>(Memory::safeMalloc(problemSize, mallocType));
    u32 const variablesCount = problemJson["nodes"].size();
    new (problem) OP::VRProblem(variablesCount, mallocType);

    // Init variables
    Variable variable(0,0);
    problem->add(&variable);
    ValueType const maxValue = static_cast<ValueType>(variablesCount - 1);
    for (u32 variableIdx = 1; variableIdx < variablesCount - 1; variableIdx += 1)
    {
        new (&variable) Variable(2, maxValue);
        problem->add(& variable);
    }
    new (&variable) Variable(1,1);
    problem->add(&variable);

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