
#include "mdd.hpp"
#include <vector>
#include <iostream>

using namespace std;

int main()
{

    int v1[] = { 1, 2, 3};
    std::vector<int> val1( v1, v1+3);
    Variable* x1 = new Variable(0, 0, val1);

    int v2[] = { 2, 1, 5, 7};
    std::vector<int> val2( v2, v2+4);
    Variable* x2 = new Variable(1, 1, val2);

    int v3[] = { 5, 2, 1, 3};
    std::vector<int> val3( v3, v3+4);
    Variable* x3 = new Variable(1, 1, val3);


    Problem problem = Problem();
    problem.add_variable( x1 );
    problem.add_variable( x2 );
    problem.add_variable( x3 );

    cout << "Problem created." << endl;

    MDD mdd = MDD( 3, 10, &problem);
    mdd.build_mdd();

    std::vector<Edge*> solution = mdd.maximize();

    for (int i = 0; i < solution.size(); i++)
    {
        Edge* edge = solution[i];
        cout << edge->get_tail()->get_layer() << " - " << edge->get_head()->get_layer() << ": " << edge->get_value() << endl;
    }


}