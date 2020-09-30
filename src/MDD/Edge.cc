#include <MDD/Edge.hh>

Edge::Edge(uint to, int value) :
    status(Valid),
    to(to),
    value(value)
{
}

bool Edge::isNotValid(Edge const & edge)
{
    return edge.status == Status::Invalid;
}





