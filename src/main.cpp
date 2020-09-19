
#include <mdd/mdd.hpp>


using namespace std;

int main()
{

    int v1[] = { 1, 2, 3};
    std::vector<int> val1( v1, v1+3);
    Variable x1 = Variable(0, 0, val1);

    int v2[] = { 2, 3};
    std::vector<int> val2( v2, v2+2);
    Variable x2 = Variable(1, 1, val2);


    Problem problem = Problem();
    problem.add_variable(x1);
    problem.add_variable(x2);

    MDD mdd = MDD( 2, 10, problem);
    mdd.build_mdd();



}