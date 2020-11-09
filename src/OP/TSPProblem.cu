#include "TSPProblem.cuh"

__host__ __device__
OP::TSPProblem::TSPProblem(unsigned int varsCount, std::byte* storage) :
    Problem(varsCount, storage),
    pickups(varsCount / 2, vars.getStorageEnd()),
    deliveries(varsCount / 2, pickups.getStorageEnd()),
    distances(varsCount * varsCount, deliveries.getStorageEnd())
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
std::size_t OP::TSPProblem::sizeofStorage(unsigned int varsCount)
{
    return Problem::sizeofStorage(varsCount) +
        StaticVector<uint16_t>::sizeofStorage(varsCount / 2) +
        StaticVector<uint16_t>::sizeofStorage(varsCount / 2) +
        RuntimeArray<uint16_t>::sizeofStorage(varsCount * varsCount);
}
__device__
uint16_t const & OP::TSPProblem::getDistance(unsigned int from, unsigned int to) const
{
    return distances[(from * vars.size) + to];
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
