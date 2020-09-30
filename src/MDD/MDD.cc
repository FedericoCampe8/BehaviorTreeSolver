#include <new>
#include <iostream>

#include <MDD/MDD.hh>

MDD::MDD(uint width, std::vector<Variable> const & vars) :
    width(width),
    layers(vars.size() + 1)
{
    for(uint i = 0; i < vars.size() + 1; i += 1)
    {
        new (&layers[i]) Layer(width, vars[i]);
    }
}

void MDD::initialize()
{
    for(uint indexLayer = 0; indexLayer < layers.size - 1; indexLayer += 1)
    {
        Layer & layer = layers[indexLayer];

        // Add one node
        layer.nodes.emplaceBack();

        // Add all outgoing edges
        for (int value = layer.minValue; value <= layer.maxValue; value += 1)
        {
            layer.edges[0].emplaceBack(0, value);
        }
    };

    // Last layer
    layers.back().nodes.emplaceBack();
}
void MDD::toGraphViz() const
{
    std::cout << "digraph G" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << std::endl;
    std::cout << "  node [shape=circle];" << std::endl;

    // Nodes
    for (uint indexLayer = 0; indexLayer < layers.size; indexLayer += 1)
    {
        std::cout << std::endl;
        std::cout << "  {" << std::endl;
        std::cout << "      rank = same;" << std::endl;

        auto const & nodes = layers[indexLayer].nodes;
        for (Node const * node = nodes.begin(); node != nodes.end(); node += 1)
        {
            std::cout << "      " << node->ID << ";" << std::endl;
        }

        std::cout << "  }" << std::endl;
    }

    // Edges
    for (uint indexLayer = 0; indexLayer < layers.size - 1; indexLayer += 1)
    {
        std::cout << std::endl;

        Layer const & layer = layers[indexLayer];
        Layer const & nextLayer = layers[indexLayer + 1];
        auto const & nodes = layer.nodes;
        auto const & nextNodes = nextLayer.nodes;

        for (uint indexNode = 0; indexNode < nodes.getSize(); indexNode += 1)
        {
            uint idParentNode = nodes[indexNode].ID;

            auto const & edges = layer.edges[indexNode];
            for(uint indexEdge = 0; indexEdge < edges.getSize(); indexEdge += 1)
            {
                Edge & edge = edges[indexEdge];
                uint idChildNode = nextNodes.at(edge.to).ID;

                std::cout << "  " << idParentNode << " -> " << idChildNode << " [label=\"" << edge.value << "\"];" << std::endl;
            }
        }
    }

    std::cout << std::endl;
    std::cout << "}" << std::endl;
}

uint MDD::getLayersCount() const
{
    return layers.size;
}
