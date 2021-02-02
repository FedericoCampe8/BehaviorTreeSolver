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
        Array<LightVector<u16>> atomicConstraintsMap;
        Array<LightVector<u16>> disjunctiveConstraints1Map;
        Array<LightVector<u16>> disjunctiveConstraints2Map;
        Array<LightVector<u16>> softAtomicConstraintsMap;

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
    u32 storagesSize = sizeof(u16) * k * atomicConstraints.getCapacity();
    u16* storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < atomicConstraintsMap.getCapacity(); index += 1)
    {
        new (atomicConstraintsMap[index]) LightVector<u16>(k, storages);
        storages = reinterpret_cast<u16*>(atomicConstraintsMap[index]->endOfStorage());
    }

    storagesSize = sizeof(u16) * k * k * disjunctiveConstraints1.getCapacity();
    storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < disjunctiveConstraints1Map.getCapacity(); index += 1)
    {
        new (disjunctiveConstraints1Map[index]) LightVector<u16>(k * k, storages);
        storages = reinterpret_cast<u16*>(disjunctiveConstraints1Map[index]->endOfStorage());
    }

    storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < disjunctiveConstraints2Map.getCapacity(); index += 1)
    {
        new (disjunctiveConstraints2Map[index]) LightVector<u16>(k * k, storages);
        storages = reinterpret_cast<u16*>(disjunctiveConstraints2Map[index]->endOfStorage());
    }

    storagesSize = sizeof(u16) * k * softAtomicConstraints.getCapacity();
    storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (unsigned int index = 0; index < softAtomicConstraints.getCapacity(); index += 1)
    {
        new (softAtomicConstraintsMap[index]) LightVector<u16>(k, storages);
        storages = reinterpret_cast<u16*>(softAtomicConstraintsMap[index]->endOfStorage());
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

    // Init variables
    ValueType const maxValue = static_cast<ValueType>(variablesCount - 1);
    Variable variable(0,maxValue);
    for (u32 variableIdx = 1; variableIdx < variablesCount; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Atomic constraints
    for (u16 atomicConstraintIdx = 0; atomicConstraintIdx < problemJson["AtomicConstraints"].size(); atomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["AtomicConstraints"][atomicConstraintIdx];
        Pair<ValueType> atomicConstraint(constraint[0],constraint[1]);
        problem->atomicConstraints.pushBack(&atomicConstraint);

        problem->atomicConstraintsMap[atomicConstraint.first]->pushBack(&atomicConstraintIdx);
        problem->atomicConstraintsMap[atomicConstraint.second]->pushBack(&atomicConstraintIdx);
    }

    // Disjunctive constraints
    u16 disjunctiveConstraint1Idx = 0;
    u16 disjunctiveConstraint2Idx = 0;
    for (u16 disjunctiveConstraintIdx = 0; disjunctiveConstraintIdx < problemJson["DisjunctiveConstraints"].size(); disjunctiveConstraintIdx += 1)
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
    for (u16 softAtomicConstraintIdx = 0; softAtomicConstraintIdx < problemJson["SoftAtomicConstraints"].size(); softAtomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["SoftAtomicConstraints"][softAtomicConstraintIdx];
        Pair<ValueType> softAtomicConstraint(constraint[0],constraint[1]);
        problem->softAtomicConstraints.pushBack(&softAtomicConstraint);

        problem->softAtomicConstraintsMap[softAtomicConstraint.first]->pushBack(&softAtomicConstraintIdx);
        problem->softAtomicConstraintsMap[softAtomicConstraint.second]->pushBack(&softAtomicConstraintIdx);
    }

    return problem;
}