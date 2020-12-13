#include "TSPProblem.cuh"

__host__ __device__
OP::TSPProblem::TSPProblem(unsigned int varsCount, std::byte* storage) :
    Problem(varsCount, storage),
    pickups(varsCount / 2, Memory::align(4,vars.storageEnd())),
    deliveries(varsCount / 2, Memory::align(4,pickups.storageEnd())),
    distances(varsCount * varsCount, Memory::align(4,deliveries.storageEnd()))
{}

__host__
void OP::TSPProblem::setStartEndLocations(unsigned int startLocation, unsigned int endLocation)
{
    this->startLocation = startLocation;
    this->endLocation = endLocation;
}

__host__
void OP::TSPProblem::addPickupDelivery(unsigned int pickup, unsigned int delivery)
{
    pickups.pushBack(pickup);
    deliveries.pushBack(delivery);
}

__host__ __device__
std::size_t OP::TSPProblem::sizeOfStorage(unsigned int varsCount)
{
    return Problem::sizeOfStorage(varsCount) +
        StaticVector<unsigned int>::sizeOfStorage(varsCount / 2) +
        StaticVector<unsigned int>::sizeOfStorage(varsCount / 2) +
        RuntimeArray<unsigned int>::sizeOfStorage(varsCount * varsCount);
}
__device__
unsigned int const & OP::TSPProblem::getDistance(unsigned int from, unsigned int to) const
{
    return distances[(from * vars.getCapacity()) + to];
}

__device__
OP::TSPProblem& OP::TSPProblem::operator=(TSPProblem const & other)
{
    Problem::operator=(other);
    this->startLocation = other.startLocation;
    this->endLocation = other.endLocation;
    this->pickups = other.pickups;
    this->deliveries = other.deliveries;
    this->distances = other.distances;

    return *this;
}
