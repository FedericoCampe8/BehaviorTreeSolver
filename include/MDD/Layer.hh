#pragma once

#include <CustomTemplateLibrary/CTL.hh>
#include <MDD/Edge.hh>
#include <MDD/Node.hh>
#include <Problem/Variable.hh>

class Layer
{
    private:
        uint const size;
        int const minValue;
        int const maxValue;
        ctl::StaticVector<Node> nodes;
        ctl::RuntimeArray<ctl::StaticVector<Edge>> edges;

    public:
        Layer(uint size, Variable const & var);

    friend class MDD;
};
