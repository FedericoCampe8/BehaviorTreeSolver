#include "VRProblem.cuh"

OP::VRProblem::VRProblem(unsigned int variablesCount, std::byte* storage) :
    Problem(variablesCount, storage),
    pickups(variablesCount / 2, variables.storageEnd()),
    deliveries(variablesCount / 2, pickups.storageEnd()),
    distances(variablesCount * variablesCount, deliveries.storageEnd())
{}

std::size_t OP::VRProblem::sizeOfStorage(unsigned int variablesCount)
{
    return
        Problem::sizeOfStorage(variablesCount) +
        StaticVector<uint8_t>::sizeOfStorage(variablesCount / 2) * 2 +
        RuntimeArray<uint16_t>::sizeOfStorage(variablesCount * variablesCount);
}

__host__ __device__
unsigned int OP::VRProblem::getDistance(unsigned int from, unsigned int to) const
{
    return distances[(from * variables.getCapacity()) + to];
}