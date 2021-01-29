#pragma once

#include <fstream>
#include <Containers/Vector.cuh>
#include <Containers/Pair.cuh>
#include <Containers/Triple.cuh>
#include <External/Nlohmann/json.hpp>
#include <Utils/Algorithms.cuh>
#include "../DP/Context.h"
#include "Problem.cuh"

namespace OP
{
    class CTWProblem : public Problem
    {
        // Members
        public:
        ValueType const b;
        ValueType const k;
        Vector<Pair<ValueType>> atomicConstraints;
        Vector<Triple<ValueType>> disjunctiveConstraints1;
        Vector<Triple<ValueType>> disjunctiveConstraints2;
        Vector<Pair<ValueType>> softAtomicConstraints;
        Array<LightVector<unsigned int>> atomicConstraintsMap;
        Array<LightVector<unsigned int>> disjunctiveConstraints1Map;
        Array<LightVector<unsigned int>> disjunctiveConstraints2Map;
        Array<LightVector<unsigned int>> softAtomicConstraintsMap;

        // Functions
        public:
        CTWProblem(unsigned int variablesCount, Memory::MallocType mallocType);
    };

    template<>
    OP::CTWProblem* parseInstance<CTWProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OP::CTWProblem::CTWProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    b(variablesCount / 2),
    k(variablesCount),
    atomicConstraints(k * k, mallocType),
    disjunctiveConstraints1(k * k * k, mallocType),
    disjunctiveConstraints2(k * k * k, mallocType),
    softAtomicConstraints(k * k, mallocType),
    atomicConstraintsMap(k, mallocType),
    disjunctiveConstraints1Map(k, mallocType),
    disjunctiveConstraints2Map(k, mallocType),
    softAtomicConstraintsMap(k, mallocType)
{
    unsigned int storagesSize = sizeof(unsigned int) * k * k;
    unsigned int* storages = reinterpret_cast<unsigned int*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < k; index += 1)
    {
        new (atomicConstraintsMap[index]) LightVector<unsigned int>(k, &storages[k * index]);
    }

    storagesSize = sizeof(unsigned int) * k * k * k;
    storages = reinterpret_cast<unsigned int*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < k; index += 1)
    {
        new (disjunctiveConstraints1Map[index]) LightVector<unsigned int>(k * k, &storages[k * k * index]);
    }

    storages = reinterpret_cast<unsigned int*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < k; index += 1)
    {
        new (disjunctiveConstraints2Map[index]) LightVector<unsigned int>(k * k, &storages[k * k * index]);
    }

    storagesSize = sizeof(unsigned int) * k * k;
    storages = reinterpret_cast<unsigned int*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < k; index += 1)
    {
        new (softAtomicConstraintsMap[index]) LightVector<unsigned int>(k, &storages[k * index]);
    }
}

OP::CTWProblem* OP::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse json
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    unsigned int const problemSize = sizeof(OP::CTWProblem);
    OP::CTWProblem* const problem = reinterpret_cast<OP::CTWProblem*>(Memory::safeMalloc(problemSize, mallocType));
    unsigned int const variablesCount = problemJson["k"];
    new (problem) OP::CTWProblem(variablesCount, mallocType);
    problem->maxBranchingFactor = variablesCount;

    // Atomic constraints
    for (unsigned int atomicConstraintIdx = 0; atomicConstraintIdx < problemJson["AtomicConstraints"].size(); atomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["AtomicConstraints"][atomicConstraintIdx];
        Pair<ValueType> atomicConstraint(constraint[0],constraint[1]);
        problem->atomicConstraints.pushBack(&atomicConstraint);

        problem->atomicConstraintsMap[atomicConstraint.first]->pushBack(&atomicConstraintIdx);
        problem->atomicConstraintsMap[atomicConstraint.second]->pushBack(&atomicConstraintIdx);
    }

    // Disjunctive constraints
    unsigned int disjunctiveConstraint1Idx = 0;
    unsigned int disjunctiveConstraint2Idx = 0;
    for (unsigned int disjunctiveConstraintIdx = 0; disjunctiveConstraintIdx < problemJson["DisjunctiveConstraints"].size(); disjunctiveConstraintIdx += 1)
    {
        auto& constraint = problemJson["DisjunctiveConstraints"][disjunctiveConstraintIdx];
        if(constraint[0] == constraint[2])
        {
            Triple<ValueType> disjunctiveConstraint(constraint[0],constraint[1],constraint[3]);
            problem->disjunctiveConstraints1.pushBack(&disjunctiveConstraint);

            problem->disjunctiveConstraints1Map[disjunctiveConstraint.first]->pushBack(&disjunctiveConstraint1Idx);
            problem->disjunctiveConstraints1Map[disjunctiveConstraint.second]->pushBack(&disjunctiveConstraint1Idx);
            problem->disjunctiveConstraints1Map[disjunctiveConstraint.third]->pushBack(&disjunctiveConstraint1Idx);

            disjunctiveConstraint1Idx += 1;
        }
        else
        {
            Triple<ValueType> disjunctiveConstraint(constraint[0],constraint[1],constraint[2]);
            problem->disjunctiveConstraints2.pushBack(&disjunctiveConstraint);

            problem->disjunctiveConstraints2Map[disjunctiveConstraint.first]->pushBack(&disjunctiveConstraint2Idx);
            problem->disjunctiveConstraints2Map[disjunctiveConstraint.second]->pushBack(&disjunctiveConstraint2Idx);
            problem->disjunctiveConstraints2Map[disjunctiveConstraint.third]->pushBack(&disjunctiveConstraint2Idx);

            disjunctiveConstraint2Idx += 1;
        }
    }

    // Soft atomic constraints
    for (unsigned int softAtomicConstraintIdx = 0; softAtomicConstraintIdx < problemJson["SoftAtomicConstraints"].size(); softAtomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["SoftAtomicConstraints"][softAtomicConstraintIdx];
        Pair<ValueType> softAtomicConstraint(constraint[0],constraint[1]);
        problem->softAtomicConstraints.pushBack(&softAtomicConstraint);

        problem->softAtomicConstraintsMap[softAtomicConstraint.first]->pushBack(&softAtomicConstraintIdx);
        problem->softAtomicConstraintsMap[softAtomicConstraint.second]->pushBack(&softAtomicConstraintIdx);
    }

    return problem;
}