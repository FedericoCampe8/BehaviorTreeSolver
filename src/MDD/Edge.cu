#include <MDD/Edge.cuh>

__device__
Edge::Edge() :
    status(Status::NotInitialized)
{}

__device__
Edge::Edge(unsigned int from, unsigned int to, int value) :
    from(from),
    to(to),
    status(Status::Active),
    value(value)
{}

__device__
bool Edge::isActive() const
{
    return status == Status::Active;
}

__device__
void Edge::deactivate()
{
    status = Status::NotActive;
}




