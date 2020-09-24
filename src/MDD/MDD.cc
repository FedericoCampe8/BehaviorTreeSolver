#include <new>
#include <vector>
#include <iostream>

#include <MDD/MDD.hh>

size_t MDD::getMemSize(std::vector<Variable> const & vars)
{
    return ctl::Vector<Layer>::getMemSize(vars.size() + 1);
}

MDD::MDD(uint maxWidth, std::vector<Variable> const & vars) :
    layers(vars.size() + 1)
{
    for(uint i = 0; i < vars.size(); i += 1)
    {
        layers.emplaceBack(maxWidth, vars[i]);
    }

    initialize();
}

void MDD::initialize()
{
    for (uint i = 0; i < layers.size() - 1; i += 1)
    {
        Layer & layer = layers.at(i);

        // Add a Node
        layer.nodes->emplaceBack();

        // Add all edges
        ctl::Vector<Edge> *edges = layer.getEdges(0);
        for (int value = layer.minValue; value <= layer.maxValue; value += 1)
        {
            edges->emplaceBack(0, value);
        }
    }

    // Last layer
    layers.back().nodes->emplaceBack();
}
void MDD::toGraphViz() const
{
    std::cout << "digraph G" << std::endl;
    std::cout << "{" << std::endl;

    std::cout << std::endl;
    std::cout << "  node [shape=circle];" << std::endl;

    // Nodes
    for (uint i = 0; i < layers.size(); i += 1)
    {
        Layer const &layer = layers.at(i);
        std::cout << std::endl;
        std::cout << "  {" << std::endl;
        std::cout << "      rank = same;" << std::endl;

        ctl::Vector<Node> const *const nodes = layer.nodes;
        for (uint i = 0; i < nodes->size(); i += 1)
        {
            std::cout << "      " << nodes->at(i).ID << ";" << std::endl;
        }

        std::cout << "  }" << std::endl;
    }

    // Edges
    for (uint i = 0; i < layers.size() - 1; i += 1)
    {
        Layer const &layer = layers.at(i);
        Layer const &nextlayer = layers.at(i + 1);

        std::cout << std::endl;

        ctl::Vector<Node> const * const nodes = layer.nodes;
        ctl::Vector<Node> const * const nodesNextlater = nextlayer.nodes;

        for (uint i = 0; i < nodes->size(); i += 1)
        {
            uint nodeID = nodes->at(i).ID;

            ctl::Vector<Edge> const * const edges = layer.getEdges(i);
            for(uint i = 0; i < edges->size(); i += 1)
            {
                Edge const edge = edges->at(i);
                std::cout << "  " << nodeID << " -> " << nodesNextlater->at(edge.to).ID << " [label=\"" << edge.value << "\"];" << std::endl;
            }
        }
    }

    std::cout << std::endl;
    std::cout << "}" << std::endl;
}
