#include <cstdlib>
#include <new>
#include <iostream>
#include <unistd.h>
#include <MDD/DAG.hh>

using namespace std;

int main()
{
    int maxWidth = 500000;
    int varsCount = 500000;

    DAG* dag = static_cast<DAG*>(malloc(DAG::getMemSize(maxWidth,varsCount)));
    new (dag) DAG(maxWidth,varsCount);

    sleep(3);

    return EXIT_SUCCESS;
}