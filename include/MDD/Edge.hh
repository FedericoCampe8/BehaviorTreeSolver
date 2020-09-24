#pragma once

#include <CustomTemplateLibrary/CTL.hh>

class Edge
{
    public:
        uint const to;
        int const value;

    public:
        Edge(uint to, int value);
};
