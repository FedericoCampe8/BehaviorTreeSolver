#pragma once

#include <MDD/Edge.hh>
#include <MDD/Node.hh>

class Layer
{
    private:
        unsigned int const maxSize;
        unsigned int const size;
        Node * const nodes;
        Edge * const edges;

    public:
        Layer(unsigned int maxSize);
};
