#include "VRProblem.cuh"

OP::VRProblem::VRProblem(unsigned int variablesCount, Memory::MallocType mallocType) :
    Problem(variablesCount, mallocType),
    pickups(variablesCount / 2, mallocType),
    deliveries(variablesCount / 2, mallocType),
    distances(variablesCount * variablesCount, mallocType)
{}

__host__ __device__
unsigned int OP::VRProblem::getDistance(unsigned int from, unsigned int to) const
{
    return *distances[(from * variables.getCapacity()) + to];
}