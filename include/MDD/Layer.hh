#pragma once

#include <CustomTemplateLibrary/CTL.hh>
#include <MDD/Edge.hh>
#include <MDD/Node.hh>
#include <Problem/Variable.hh>

class Layer
{

    private:
        int const minValue;
        int const maxValue;
        size_t const edgesMemSize;
        ctl::Vector<Node> * const nodes;
        ctl::Vector<Edge> * const edges;

    public:
        Layer(uint maxSize, Variable const & var);

        ctl::Vector<Edge> * getEdges(uint index) const;

    private:
        ctl::Vector<Node> * const mallocNodes(uint maxSize);
        ctl::Vector<Edge> * const mallocEdges(uint maxSize);

    friend class MDD;
};
