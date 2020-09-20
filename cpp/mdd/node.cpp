#include "node.hpp"

Node::Node( Variable* variable, uint32_t layer) 
{ 
    this->variable = variable;
    this->layer = layer;

    // Null variable indicates leaf node of mdd...only incoming edges from last variable of problem.
    if (variable != NULL) {
        for (int i = 0; i < variable->get_available_values().size(); ++i) {
            this->available_values.push_back( variable->get_available_values()[i] );
        }
    }

} 

std::vector<int>* Node::get_values()
{
    return &available_values;
}

uint32_t Node::get_layer()
{
    return layer;
}


void Node::add_in_edge( Edge* edge ) 
{ 
    this->inEdges.push_back( edge ); 
}

void Node::add_out_edge( Edge* edge ) 
{ 
    this->outEdges.push_back( edge); 
}


void Node::remove_in_edge( Edge* edge ) 
{ 
    for (int k = 0; k < inEdges.size(); ++k) {
        if (edge == inEdges[k] ) {
            inEdges.erase(inEdges.begin()+k); 
            break;
        }
    }
}
    
void Node::remove_out_edge( Edge* edge ) 
{ 
    for (int k = 0; k < outEdges.size(); ++k) {
        if (edge == outEdges[k] ) {
            outEdges.erase(outEdges.begin()+k); 
            break;
        }
    }
}


std::vector<Edge*> Node::get_out_edges() 
{ 
    return outEdges; 
}

std::vector<Edge*> Node::get_in_edges() 
{ 
    return inEdges; 
}

Variable* Node::get_variable() 
{ 
    return variable; 
}

void Node::set_optimization_value( float opt_value ) 
{ 
    optimization_value = opt_value; 
}

float Node::get_optimization_value() 
{ 
    return optimization_value; 
}

void Node::set_selected_edge( Edge* edge ) 
{ 
    selected_edge = edge; 
}

Edge* Node::get_selected_edge() 
{ 
    return selected_edge; 
}

bool Node::is_leaf()
{
    return variable == NULL;
}
