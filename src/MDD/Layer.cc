#include <cstddef>
#include <cstdlib>
#include <new>

#include <MDD/Layer.hh>

Layer::Layer(uint size, Variable const & var) :
    size(size),
    minValue(var.minValue),
    maxValue(var.maxValue),
    nodes(size),
    edges(size)
{
    uint edgesPerNode = maxValue - minValue + 1;
    size_t edgesMemSize  = size * edgesPerNode * sizeof(Edge);
    Edge * edgesMem = static_cast<Edge*>(malloc(edgesMemSize));
    for (uint i = 0; i < size; i += 1)
    {
        new (&edges[i]) ctl::StaticVector<Edge>(edgesPerNode, edgesMem + edgesPerNode * i);
    }
}

