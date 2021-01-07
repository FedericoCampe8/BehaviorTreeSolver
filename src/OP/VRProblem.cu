#include <fstream>
#include <External/NlohmannJson.hpp>
#include "VRProblem.cuh"

OP::VRProblem::VRProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    start(0),
    end(0),
    pickups(variablesCount / 2, mallocType),
    deliveries(variablesCount / 2, mallocType),
    distances(variablesCount * variablesCount, mallocType)
{}

__host__ __device__
unsigned int OP::VRProblem::getDistance(ValueType from, ValueType to) const
{
    return *distances[(from * variables.getCapacity()) + to];
}

OP::VRProblem * OP::VRProblem::parseGrubHubInstance(char const * problemFileName, Memory::MallocType mallocType)
{
    // Parse instance
    std::ifstream problemFile(problemFileName);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    unsigned int const memorySize = sizeof(OP::VRProblem);
    std::byte* const memory = safeMalloc(memorySize, mallocType);
    unsigned int const variablesCount = problemJson["nodes"].size();
    OP::VRProblem* const problem  = new (memory) OP::VRProblem(variablesCount, mallocType);
    problem->maxBranchingFactor = (variablesCount - 1) - 2 + 1;
    problem->start = 0;
    problem->end = 1;

    // Init variables
    for(unsigned int variableIdx = 0; variableIdx < variablesCount; variableIdx += 1)
    {
        if(variableIdx == 0)
        {
            new (problem->variables[variableIdx]) OP::Variable(0, 0);
        }
        else if (variableIdx == variablesCount - 1)
        {
            new (problem->variables[variableIdx]) OP::Variable(1, 1);
        }
        else
        {
            new(problem->variables[variableIdx]) OP::Variable(2, variablesCount - 1);
        }
    }

    // Init pickups and deliveries
    for(OP::ValueType pickup = 2; pickup < variablesCount; pickup += 2)
    {
        problem->pickups.pushBack(&pickup);
        OP::ValueType const delivery = pickup + 1;
        problem->deliveries.pushBack(&delivery);
    }

    // Init distances
    for(unsigned int from = 0; from < variablesCount; from += 1)
    {
        for(unsigned int to = 0; to < variablesCount; to += 1)
        {
            *problem->distances[(from * variablesCount) + to] = problemJson["edges"][from][to];
        }
    }

    return problem;
}