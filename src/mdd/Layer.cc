#include <cstdlib>

#include <MDD/Layer.hh>

using namespace std;

Layer::Layer(unsigned int maxSize) :
    maxSize(maxSize),
    size(0),
    nodes(static_cast<Node*>(aligned_alloc(alignof(Node), sizeof(Node) * maxSize))),
    edges(static_cast<Edge*>(aligned_alloc(alignof(Edge), sizeof(Node) * maxSize * maxSize)))
{
}