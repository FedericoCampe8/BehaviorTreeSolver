#pragma once

#include <cstddef>

#include <Extra/Extra.hh>
#include <MDD/Edge.hh>
#include <MDD/Node.hh>

class Layer
{
    private:
        uint const size;
        int const minValue;
        int const maxValue;
        Extra::Containers::RestrainedVector<Node> nodes;
        Edge * const storageEdges;
        Extra::Containers::RestrainedArray<Extra::Containers::RestrainedVector<Edge>> edges;

    public:
        __device__ Layer(uint size, int minValue, int maxValue);
        __device__ ~Layer();
    private:
        __device__ Edge * getStorageEdges() const;

    friend class MDD;
};
