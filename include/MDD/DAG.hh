#pragma once

#include <cstdlib>

#include <MDD/Layer.hh>

class DAG
{
    private:
        int const layerMaxSize;
        int const layersCount;
        Layer layers[];

    public:
        static size_t getMemSize(int maxWidth, int varsCount);

        DAG(int maxWidth, int varsCount);
};