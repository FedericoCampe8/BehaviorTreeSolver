#include "Edge.cuh"

__device__
MDD::Edge::Edge() :
    status(Status::NotActive)
{}

__device__
MDD::Edge::Edge(unsigned int to) :
    status(Status::Active),
    to(to)
{}

__device__
bool MDD::Edge::isActive(Edge const & edge)
{
    return edge.status == Status::Active;
}

__device__
void MDD::Edge::reset(MDD::Edge& edge)
{
    edge.status = Status::NotActive;
}
