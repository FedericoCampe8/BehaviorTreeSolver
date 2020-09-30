#include <new>
#include <iostream>
#include <fstream>

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

void MDD::toGraphViz(std::string const & nameFileGv) const
{
    std::ofstream fileGv;
    fileGv.open(nameFileGv, std::ios::out | std::ios::trunc);
    assert(fileGv.is_open());

    fileGv << "digraph G" << std::endl;
    fileGv << "{" << std::endl;
    fileGv << std::endl;
    fileGv << "  node [shape=circle];" << std::endl;

    // Nodes
    for (uint indexLayer = 0; indexLayer < layers.size; indexLayer += 1)
    {
        fileGv << std::endl;
        fileGv << "  {" << std::endl;
        fileGv << "      rank = same;" << std::endl;

        auto const & nodes = layers[indexLayer].nodes;
        for (Node const * node = nodes.begin(); node != nodes.end(); node += 1)
        {
            fileGv << "      " << node->ID << ";" << std::endl;
        }

        fileGv << "  }" << std::endl;
    }

    // Edges
    for (uint indexLayer = 0; indexLayer < layers.size - 1; indexLayer += 1)
    {
        fileGv << std::endl;

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

                fileGv << "  " << idParentNode << " -> " << idChildNode << " [label=\"" << edge.value << "\"];" << std::endl;
            }
        }
    }

    fileGv << std::endl;
    fileGv << "}" << std::endl;
    fileGv.close();

}


void MDD::DFS(uint indexLayer, uint indexNode, ctl::StaticVector<int> & labels, bool print) const
{
    if(indexLayer == layers.size - 1)
    {
        if (print)
        {
            std::cout << labels[0];

            for (uint i = 1; i < indexLayer; i += 1)
            {
                std::cout << "," << labels[i];
            }

            std::cout << std::endl;
        }
    }
    else
    {
        auto const &edges = layers[indexLayer].edges[indexNode];
        for (uint indexEdge = 0; indexEdge < edges.getSize(); indexEdge += 1)
        {
            Edge const & edge = edges[indexEdge];
            labels.emplaceBack(edge.value);
            DFS(indexLayer + 1,  edge.to, labels, print);
            labels.popBack();
        }

    }
}

uint MDD::getLayersCount() const
{
    return layers.size;
}
