#include <cstdlib>
#include <vector>

#include <MDD/MDD.hh>
#include <Problem/Variable.hh>


int main()
{

    // Variables
    unsigned int varsCount = 10;
    Variable v({0,10});
    std::vector<Variable> vars(varsCount, v);

    // MDD
    unsigned int maxWidth = 10;
    MDD* dag = static_cast<MDD*>(malloc(MDD::getMemSize(vars)));
    new (dag) MDD(maxWidth, vars);

    dag->toGraphViz();

    return EXIT_SUCCESS;
}