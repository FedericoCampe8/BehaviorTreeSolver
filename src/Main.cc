#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>

#include <DP/AllDifferent.hh>
#include <MDD/MDD.hh>
#include <Problem/Variable.hh>

using namespace std::chrono;

int main()
{

    // Variables
    unsigned int varsCount = 4;
    Variable v({0,static_cast<int>(varsCount) - 1});
    std::vector<Variable> vars(varsCount, v);

    // MDD
    unsigned int width = AllDifferent::getOptimalLayerWidth(varsCount);
    std::cout << "[INFO] MDD with " << width << std::endl;

    auto start = high_resolution_clock::now();
    MDD * const mdd = new MDD(width, vars);
    mdd->initialize();
    mdd->separate<AllDifferent>();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "[INFO] MDD created in " << duration.count() << " ms" << std::endl;

    // DFS
    start = high_resolution_clock::now();
    ctl::StaticVector<int> labels(mdd->getLayersCount());
    mdd->DFS(0, 0,labels, true);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    std::cout << "[INFO] DFS performed in " << duration.count() << " ms" << std::endl;

    // GraphViz
    std::string nameFileGv = "mdd.gv";
    mdd->toGraphViz(nameFileGv);
    std::cout << "[INFO] GraphViz saved in " << nameFileGv << std::endl;

    return EXIT_SUCCESS;
}