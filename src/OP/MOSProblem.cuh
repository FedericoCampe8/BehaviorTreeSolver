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
    class MOSProblem : public Problem
    {
        // Members

        public:
        OP::ValueType const clients;
        OP::ValueType const products;
        Array<u16> orders;

        // Functions
        public:
        MOSProblem(u32 clients, u32 products, Memory::MallocType mallocType);
        __host__ __device__ inline DP::CostType getOrder(ValueType client, ValueType product) const;
    };

    template<>
    OP::MOSProblem* parseInstance<MOSProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::MOSProblem::MOSProblem(u32 clients, u32 products, Memory::MallocType mallocType) :
    Problem(products, mallocType),
    clients(clients),
    products(products),
    orders(clients * products, mallocType)
{}

__host__ __device__
DP::CostType OP::MOSProblem::getOrder(ValueType client, ValueType product) const
{
    return *orders[(client * products) + product];
}

OP::MOSProblem* OP::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse json
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OP::MOSProblem);
    OP::MOSProblem* const problem = reinterpret_cast<OP::MOSProblem*>(Memory::safeMalloc(problemSize, mallocType));
    OP::ValueType const clients = problemJson["clients"];
    OP::ValueType const products = problemJson["products"];
    new (problem) OP::MOSProblem(clients, products, mallocType);

    // Init variables
    Variable variable(0, products - 1);
    for (OP::ValueType variableIdx = 0; variableIdx < products; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Init orders
    for (OP::ValueType client = 0; client < clients; client += 1)
    {
        for (OP::ValueType product = 0; product < products; product += 1)
        {
            *problem->orders[(client * products) + product] = problemJson["orders"][client][product];
        }
    }
    return problem;
}