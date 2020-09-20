#include "problem.hpp"

void Problem::add_variable(Variable var)
{ 
    variables.push_back( var ); 
}

std::vector<Variable> Problem::get_variables() 
{ 
    return variables; 
} 
