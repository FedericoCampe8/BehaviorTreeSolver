#include <cstddef>
#include <cstdlib>
#include <new>

#include <MDD/Layer.hh>

Layer::Layer(uint maxSize, Variable const & var) :
    minValue(var.minValue),
    maxValue(var.maxValue),
    edgesMemSize(ctl::Vector<Edge>::getMemSize(maxValue - minValue + 1)),
    nodes(Layer::mallocNodes(maxSize)),
    edges(Layer::mallocEdges(maxSize))

{
    new (nodes) ctl::Vector<Node>(maxSize);


    std::byte * edgesMem = reinterpret_cast<std::byte*>(edges);
    for (uint i = 0; i < maxSize; i += 1)
    {
        size_t offset = i * edgesMemSize;
        new (edgesMem + offset) ctl::Vector<Edge>(maxValue - minValue + 1);
    }
}

ctl::Vector<Edge> * Layer::getEdges(uint index) const
{
    std::byte * edgesMem = reinterpret_cast<std::byte*>(edges);
    size_t offset = index * edgesMemSize;
    return reinterpret_cast<ctl::Vector<Edge>*>(edgesMem + offset);
}

ctl::Vector<Node> * const Layer::mallocNodes(uint maxSize)
{
    size_t nodesMemSize = ctl::Vector<Node>::getMemSize(maxSize);

    void * nodesMem = malloc(nodesMemSize);
    assert(nodesMem != nullptr);

    return static_cast<ctl::Vector<Node>*>(nodesMem);
}

 ctl::Vector<Edge> * const Layer::mallocEdges(uint maxSize)
{
    void * edgesMem = malloc(edgesMemSize * maxSize);
    assert(edgesMem != nullptr);

    return static_cast<ctl::Vector<Edge>*>(edgesMem);
}

