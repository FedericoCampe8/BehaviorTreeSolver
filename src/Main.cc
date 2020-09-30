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
    unsigned int varsCount = 10;
    Variable v({0,static_cast<int>(varsCount) - 1});
    std::vector<Variable> vars(varsCount, v);

    // MDD
    unsigned int width = 300;

    auto start = high_resolution_clock::now();
    MDD * const mdd = new MDD(width, vars);
    mdd->initialize();
    mdd->separate<AllDifferent>();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "MDD created in " << duration.count() << " ms" << std::endl;

    //mdd->toGraphViz();

    return EXIT_SUCCESS;
}