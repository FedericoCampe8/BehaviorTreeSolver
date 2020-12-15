#include "VRP.cuh"

OP::VRP::VRP(unsigned int variablesCount, std::byte* storage) :
    variables(variablesCount, storage),
    pickups(variablesCount / 2, variables.storageEnd()),
    deliveries(variablesCount / 2, pickups.storageEnd()),
    distances(variablesCount * variablesCount, deliveries.storageEnd())
{}

void OP::VRP::addPickupDelivery(Variable::ValueType pickup, Variable::ValueType delivery)
{
    pickups.pushBack(pickup);
    deliveries.pushBack(delivery);
}

std::size_t OP::VRP::sizeOfStorage(unsigned int variablesCount)
{
    return
        StaticVector<Variable::ValueType>::sizeOfStorage(variablesCount) +
        StaticVector<Variable::ValueType>::sizeOfStorage(variablesCount / 2) * 2 +
        RuntimeArray<DistanceType>::sizeOfStorage(variablesCount * variablesCount);
}

__host__ __device__
OP::VRP::DistanceType OP::VRP::getDistance(Variable::ValueType from, Variable::ValueType to) const
{
    return distances[(from * variables.getCapacity()) + to];
}