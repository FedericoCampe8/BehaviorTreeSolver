#include "mdd.hpp"



MDD::MDD(int32_t vars, int32_t width, Problem* problem)
{
    //TODO: max width not implemented in constraint
    this->num_variables = vars;
    this->max_width = width;
    this->problem = problem;
    this->nodes_per_layer = std::vector< std::vector<Node*> > (vars+1);

    cout << "MDD initialized." << endl;
}


void MDD::build_mdd() 
{
    std::vector<Variable*> variables = get_problem()->get_variables();
    int total_layers = variables.size();

    Variable* current_var = get_problem()->get_variables().at(0);
    Node* current_node = new Node( current_var, 0 );

    nodes_per_layer.at(0).push_back( current_node );

    // Create initial mdd
    for (int i = 0; i < total_layers; i++)
    {
        Node* next_node = expand_node( current_node );
        current_node = next_node;
        nodes_per_layer.at( current_node->get_layer() ).push_back( current_node );
    }
    

    // Enforce alldiff constraint
    for (int i = 0; i < total_layers; i++) {
        for (int k = 0; k < nodes_per_layer[i].size(); k++ ) {
            Node* node = nodes_per_layer[i][k];

            enforce_alldifff_sequential( node );
        }
    }

}

Node* MDD::expand_node( Node* node )
{
    std::vector<int>* values = node->get_values();
    Variable* next_var = NULL;
    if ( node->get_layer()+1 < get_problem()->get_variables().size() ) {
        next_var = get_problem()->get_variables().at( node->get_layer()+1 );
    }

    Node* next_node = new Node( next_var, node->get_layer()+1 );
    
    for (int i = 0; i < values->size(); ++i) {
        
        Edge* edge = new Edge( node,  next_node, values->at(i) );

        next_node->add_in_edge( edge );
        node->add_out_edge( edge );
    }

    return next_node;
}

void MDD::enforce_alldifff_sequential( Node *node )
{
    std::vector<Node*> children;
    // Find all children nodes 
    for (int k = 0; k < node->get_out_edges().size(); k++) {
        Node* next_node = node->get_out_edges()[k]->get_head();

        // Check that node hasn't already been added
        bool node_found = false;
        for (int i = 0; i < children.size(); ++i) {
            Node* child = children[i];
            if (child == next_node) {
                node_found = true;
                break;
            }
        }

        if (node_found == false && next_node->is_leaf() == false) {
            children.push_back( next_node );
        }
    }


    //Enforce all diff contraint by splitting nodes
    for (int n = 0; n < children.size(); ++n)
    {
        Node* next_node = children[n];
        std::vector<int>* available_values_tail = node->get_values();
        std::vector<int>* available_values_head = next_node->get_values();
        std::vector<int> conflicting_values;

        // Find all conflicting values
        for (int i = 0; i < available_values_tail->size(); ++i) {
            
            std::vector<int>::iterator it;
            it = std::find(available_values_head->begin(), available_values_head->end(), available_values_tail->at(i));
            if (it != available_values_head->end()) {
                conflicting_values.push_back( *it );
            }
        }


        for (int edge_idx = 0; edge_idx < node->get_out_edges().size(); ++edge_idx) {
            Edge* edge = node->get_out_edges()[edge_idx];
            std::vector<int>::iterator it;

            int conflicting_value;
            int position = -1;

            // If outgoing edge is in conflicting values find its position
            for (int k = 0; k < conflicting_values.size(); ++k) {
                if (conflicting_values[k] == edge->get_value()) {
                    conflicting_value = edge->get_value();
                    position = k;
                    break;
                }

            }
            
            if (position > -1) {
                // edge points to conflicting value, so split node
                Node* new_node = new Node( next_node->get_variable(), next_node->get_layer() );
                this->nodes_per_layer[ next_node->get_layer() ].push_back( new_node );

                std::vector<int>* new_available_values = new_node->get_values();
                new_available_values->erase( new_available_values->begin() + position );


                // Need to remove nodes with only invalid outgoing edges...
                // TODO
                // if (next_node->get_out_edges().size() == 1 && next_node->get_out_edges()[0]->get_value() == edge->get_value()) {
                //     continue;
                // }

                // Move incoming conflicting edge from next node to splitting node
                next_node->remove_in_edge( edge );
                new_node->add_in_edge( edge );
                edge->set_head( new_node );

                // Copy outgoing edges from next node to splitting node
                for (int out_edge_idx = 0; out_edge_idx < next_node->get_out_edges().size(); ++out_edge_idx) {
                    Edge* edge_to_copy = next_node->get_out_edges()[out_edge_idx];
                    if (edge_to_copy->get_value() != conflicting_value) {
                        Edge* new_out_edge = new Edge( new_node,  edge_to_copy->get_head(), edge_to_copy->get_value() );
                        new_node->add_out_edge( new_out_edge );
                        edge_to_copy->get_head()->add_in_edge( new_out_edge );
                    }
                }

            }
        }
    }

}


std::vector<Edge*> MDD::maximize()
{
    std::stack<Node*> nodes_to_expand;
    std::vector<Node*> visited;
    Node* root = nodes_per_layer[0][0];
    root->set_optimization_value( 0.0 );
    nodes_to_expand.push( root );

    while (nodes_to_expand.size() > 0) {
        Node* currentNode = nodes_to_expand.top();
        nodes_to_expand.pop();

        // Many edges can point to the same node
        for (int i = 0; i < currentNode->get_out_edges().size(); ++i) {
            Edge* edge = currentNode->get_out_edges().at(i);
            Node* next_node = edge->get_head();

            bool node_found = false;
            for (int k = 0; k < visited.size(); k++) {
                Node* tmp = visited[k];
                if ( tmp == next_node) {
                    node_found = true;
                    break;
                }
            }

            if (node_found == false) {
                visited.push_back( next_node );
            }

            if ( node_found == false ) {
                // If node does not exists, insert it.
                nodes_to_expand.push( next_node );
                next_node->set_optimization_value(  currentNode->get_optimization_value() + edge->get_value() );
                next_node->set_selected_edge( edge );
            } else {
                // If it does, update its partial solution
                float candidate_value = currentNode->get_optimization_value() + edge->get_value();
                if ( candidate_value > next_node->get_optimization_value() ) {
                    next_node->set_optimization_value( candidate_value );
                    next_node->set_selected_edge( edge );
                }

            }

        }

    }

    std::vector<Edge*> solution;
    Node* node = nodes_per_layer[nodes_per_layer.size()-1][0]; //Start with leaf and trace backwards
    while (node->get_selected_edge() != NULL ) {
        solution.push_back( node->get_selected_edge() );
        node = node->get_selected_edge()->get_tail();
    }

    return solution;
}

std::vector<Edge*> MDD::minimize()
{

}
