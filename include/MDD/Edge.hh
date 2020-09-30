#pragma once

#include <CustomTemplateLibrary/CTL.hh>

class Edge
{
    public:
        enum Status {Valid, Invalid};

        Status status;
        uint to;
        int  value;

    public:
        Edge(uint to, int value);

        static bool isNotValid(Edge const & edge);
};
