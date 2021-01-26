#pragma once

#include <fstream>
#include <Containers/Vector.cuh>
#include <Containers/Pair.cuh>
#include <External/Nlohmann/json.hpp>
#include "../DP/Context.h"
#include "Problem.cuh"

namespace OP
{
    class CTWProblem : public Problem
    {
        // Members
        public:
        ValueType b;
        ValueType k;
        Vector<Pair<ValueType>> atomicConstraints;
        Vector<Pair<ValueType>> softAtomicConstraints;
        Vector<Pair<Pair<ValueType>>> disjunctiveConstraints;

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
    softAtomicConstraints(k * k, mallocType),
    disjunctiveConstraints(k * k * k * 2, mallocType)
{}

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
    problem->maxBranchingFactor = (variablesCount - 1) - 2 + 1;

    // Atomic constraints
    for (unsigned int atomicConstraintIdx = 0; atomicConstraintIdx < problemJson["AtomicConstraints"].size(); atomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["AtomicConstraints"][atomicConstraintIdx];
        Pair<OP::ValueType> atomicConstraint(constraint[0],constraint[1]);
        problem->atomicConstraints.pushBack(&atomicConstraint);
    }

    // Disjunctive constraints
    for (unsigned int disjunctiveConstraintIdx = 0; disjunctiveConstraintIdx < problemJson["DisjunctiveConstraints"].size(); disjunctiveConstraintIdx += 1)
    {
        auto& constraint = problemJson["DisjunctiveConstraints"][disjunctiveConstraintIdx];
        Pair<OP::ValueType> disjunctiveConstraint0(constraint[0], constraint[1]);
        Pair<OP::ValueType> disjunctiveConstraint1(constraint[2], constraint[3]);
        Pair<Pair<OP::ValueType>> disjunctiveConstraint(disjunctiveConstraint0,disjunctiveConstraint1);
        problem->disjunctiveConstraints.pushBack(&disjunctiveConstraint);
    }

    // Soft atomic constraints
    for (unsigned int softAtomicConstraintIdx = 0; softAtomicConstraintIdx < problemJson["SoftAtomicConstraints"].size(); softAtomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["SoftAtomicConstraints"][softAtomicConstraintIdx];
        Pair<OP::ValueType> softAtomicConstraint(constraint[0], constraint[1]);
        problem->softAtomicConstraints.pushBack(&softAtomicConstraint);
    }

    return problem;
}