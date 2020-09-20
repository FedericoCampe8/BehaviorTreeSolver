
#include "mdd.hpp"
#include <vector>
#include <iostream>

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

    cout << "Problem created." << endl;

    MDD mdd = MDD( 2, 10, problem);
    mdd.build_mdd();

    std::vector<Edge> solution = mdd.maximize();

    for (int i = 0; i < solution.size(); ++i)
    {
        Edge edge = solution.at(i);
        cout << edge.get_tail()->get_layer() << " - " << edge.get_head()->get_layer() << ": " << edge.get_value() << endl;
    }


}