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
        ValueType start;
        ValueType end;
        Vector<ValueType> pickups;
        Vector<ValueType> deliveries;
        Array<DP::CostType> distances;

        // Functions
        public:
        VRProblem(unsigned int variablesCount, Memory::MallocType mallocType);
        __host__ __device__ inline DP::CostType getDistance(ValueType from, ValueType to) const;
    };

    template<>
    OP::VRProblem* parseInstance<OP::VRProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::VRProblem::VRProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    pickups(variablesCount / 2, mallocType),
    deliveries(variablesCount / 2, mallocType),
    distances(variablesCount * variablesCount, mallocType)
{}

__host__ __device__
unsigned int OP::VRProblem::getDistance(ValueType from, ValueType to) const
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

    unsigned int const problemSize = sizeof(OP::VRProblem);
    OP::VRProblem* const problem = reinterpret_cast<OP::VRProblem*>(Memory::safeMalloc(problemSize, mallocType));
    unsigned int const variablesCount = problemJson["nodes"].size();
    new (problem) OP::VRProblem(variablesCount, mallocType);
    problem->maxBranchingFactor = (variablesCount - 1) - 2 + 1;
    problem->start = 0;
    problem->end = 1;

    // Init variables
    for (unsigned int variableIdx = 0; variableIdx < variablesCount; variableIdx += 1)
    {
        if (variableIdx == 0)
        {
            new (problem->variables[variableIdx]) OP::Variable(0, 0);
        }
        else if (variableIdx == variablesCount - 1)
        {
            new (problem->variables[variableIdx]) OP::Variable(1, 1);
        }
        else
        {
            new (problem->variables[variableIdx]) OP::Variable(2, static_cast<ValueType>(variablesCount - 1));
        }
    }

    // Init pickups and deliveries
    for (OP::ValueType pickup = 2; pickup < variablesCount; pickup += 2)
    {
        problem->pickups.pushBack(& pickup);
        OP::ValueType const delivery = pickup + 1;
        problem->deliveries.pushBack(& delivery);
    }

    // Init distances
    for (unsigned int from = 0; from < variablesCount; from += 1)
    {
        for (unsigned int to = 0; to < variablesCount; to += 1)
        {
            *problem->distances[(from * variablesCount) + to] = problemJson["edges"][from][to];
        }
    }

    return problem;
}