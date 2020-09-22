#include <MDD/DAG.hh>

#include <new>
#include <iostream>
#include <algorithm>
#include <errno.h>
#include <string.h>

#include <MDD/DAG.hh>

using namespace std;

size_t DAG::getMemSize(int maxWidth, int varsCount)
{
    return sizeof(DAG) + (sizeof(Layer) * (varsCount + 1));
}

DAG::DAG(int maxWidth, int varsCount) :
    layerMaxSize(maxWidth),
    layersCount(varsCount + 1)
{
    for(int i = 0; i < layersCount; i += 1)
    {
        new (&layers[i]) Layer(layerMaxSize);
    }
}