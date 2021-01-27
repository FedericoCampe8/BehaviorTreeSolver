#pragma once

#include <fstream>
#include <Containers/Vector.cuh>
#include <Containers/Pair.cuh>
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
        Array<LightVector<ValueType>> atomicConstraints;
        Array<LightVector<ValueType>> disjunctiveConstraints1;
        Array<LightVector<ValueType>> disjunctiveConstraints2;
        Array<LightVector<ValueType>> softAtomicConstraints;

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
    atomicConstraints(k, mallocType),
    disjunctiveConstraints1(k, mallocType),
    disjunctiveConstraints2(k, mallocType),
    softAtomicConstraints(k, mallocType)
{
    unsigned int const storagesSize = sizeof(ValueType) * k * k * 4;
    ValueType* storages = reinterpret_cast<ValueType*>(Memory::safeMalloc(storagesSize, mallocType));
    ValueType* atomicConstraintsStorages = &storages[k * k * 0];
    ValueType* disjunctiveConstraints1Storages = &storages[k * k * 1];
    ValueType* disjunctiveConstraints2Storages = &storages[k * k * 2];
    ValueType* softAtomicConstraintsStorages = &storages[k * k * 3];

    for (unsigned int index = 0; index < k; index += 1)
    {
        new (atomicConstraints[index]) LightVector<ValueType>(k, &atomicConstraintsStorages[k * index]);
        new (disjunctiveConstraints1[index]) LightVector<ValueType>(k, &disjunctiveConstraints1Storages[ k * index]);
        new (disjunctiveConstraints2[index]) LightVector<ValueType>(k, &disjunctiveConstraints2Storages[k * index]);
        new (softAtomicConstraints[index]) LightVector<ValueType>(k, &softAtomicConstraintsStorages[k * index]);
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
        problem->atomicConstraints[constraint[0]]->pushBack(&constraint[1]);
    }

    // Disjunctive constraints
    for (unsigned int disjunctiveConstraintIdx = 0; disjunctiveConstraintIdx < problemJson["DisjunctiveConstraints"].size(); disjunctiveConstraintIdx += 1)
    {
        auto& constraint = problemJson["DisjunctiveConstraints"][disjunctiveConstraintIdx];
        if(constraint[0] == constraint[2])
        {
            problem->disjunctiveConstraints1[constraint[0]]->pushBack(&constraint[1]);
            problem->disjunctiveConstraints1[constraint[0]]->pushBack(&constraint[3]);
        }
        else
        {
            problem->disjunctiveConstraints2[constraint[0]]->pushBack(&constraint[1]);
            problem->disjunctiveConstraints2[constraint[0]]->pushBack(&constraint[2]);
        }
    }

    // Soft atomic constraints
    for (unsigned int softAtomicConstraintIdx = 0; softAtomicConstraintIdx < problemJson["SoftAtomicConstraints"].size(); softAtomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["AtomicConstraints"][softAtomicConstraintIdx];
        problem->softAtomicConstraints[constraint[0]]->pushBack(&constraint[1]);
    }

    return problem;
}