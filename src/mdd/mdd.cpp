#include <algorithm>
#include <mdd/mdd.hpp>



MDD::MDD(int32_t vars, int32_t width, Problem problem)
{
    //TODO: max width not implemented in constraint
    this->num_variables = vars;
    this->max_width = width;
    this->problem = &problem;
    this->nodes_per_layer = std::vector< std::vector<Node> > (vars);

}


void MDD::build_mdd() 
{
    std::vector<Variable> variables = get_problem()->get_variables();
    int total_layers = variables.size();

    Variable current_var = get_problem()->get_variables().at(0);
    Node current_node = Node( &current_var, 0 );
    Node* current_node_ptr = &current_node;

    nodes_per_layer.at(0).push_back( current_node );

    // Create initial mdd
    for (int i = 0; i < total_layers-1; ++i)
    {
        Node* next_node = expand_node( current_node_ptr );
        current_node_ptr = next_node;
        nodes_per_layer.at( current_node_ptr->get_layer() ).push_back( *current_node_ptr );
    }
    

    // Enforce alldiff constraint
    for (int i = 0; i < total_layers; ++i) {
        for (int k = 0; k < nodes_per_layer.at(i).size(); ++i ) {
            Node* node = &nodes_per_layer.at(i).at(k);

            enforce_alldifff_sequential( node );
        }
    }

}

Node* MDD::expand_node( Node* node )
{
    std::vector<int>* values = node->get_values();
    Variable next_var = get_problem()->get_variables().at( node->get_layer()+1 );
    Node next_node = Node( &next_var, node->get_layer()+1 );
    Node* next_node_prt = &next_node;
    
    for (int i = 0; i < values->size(); ++i) {
        
        Edge edge = Edge( node,  &next_node, values->at(i) );

        next_node.add_in_edge( &edge );
        node->add_out_edge( &edge );
    }

    return next_node_prt;
}

void MDD::enforce_alldifff_sequential( Node *node )
{
    std::vector<Node> children;
    // Find all children nodes 
    for (int k = 0; k < node->get_out_edges().size(); k++) {
        Node* next_node = node->get_out_edges()[k].get_head();

        // Check that node hasn't already been added
        bool node_found = false;
        for (int i = 0; i < children.size(); ++i) {
            Node child = children[i];
            if (&child == next_node) {
                node_found = true;
                break;
            }
        }

        if (node_found == false) {
            children.push_back( *next_node );
        }
    }


    //Enforce all diff contraint by splitting nodes
    for (int n = 0; children.size(); ++n)
    {
        Node* next_node = &children.at(n);
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
            Edge edge = node->get_out_edges()[edge_idx];
            Edge* edge_ptr = &edge;
            std::vector<int>::iterator it;

            int value;
            int position = -1;

            // If outgoing edge is in conflicting values find its position
            for (int k = 0; k < conflicting_values.size(); ++k) {
                if (conflicting_values[k] == edge_ptr->get_value()) {
                    value = edge_ptr->get_value();
                    position = k;
                    break;
                }

            }
            
            if (position > -1) {
                // edge points to conflicting value, so split node
                Node new_node = Node( next_node->get_variable(), next_node->get_layer() );
                this->nodes_per_layer[ next_node->get_layer() ].push_back( new_node );

                std::vector<int>* new_available_values = new_node.get_values();
                new_available_values->erase( new_available_values->begin() + position );

                // As I remove edges, some of them might be changing in the vector
                int position = -1;
                for (int k = 0; k < node->get_out_edges().size(); ++k) {
                    if (edge_ptr == &node->get_out_edges()[k] ) {
                        position = k;
                        break;
                    }
                }

                // If there is only one outgoing edge and that edge is invalid with new node, stop.
                if (next_node->get_out_edges().size() == 1 && next_node->get_out_edges()[0].get_value() == edge_ptr->get_value()) {
                    continue;
                }

                // Move incoming conflicting edge from next node to splitting node
                next_node->remove_in_edge( position );
                new_node.add_in_edge( edge_ptr );
                edge_ptr->set_head( &new_node );

                // Copy outgoing edges from next node to splitting node
                for (int out_edge_idx = 0; out_edge_idx < next_node->get_out_edges().size(); ++out_edge_idx) {
                    Edge copy_edge = next_node->get_out_edges()[out_edge_idx];
                    Edge new_out_edge = Edge( &new_node, copy_edge.get_head(), copy_edge.get_value() );
                    new_node.add_out_edge( &new_out_edge );
                }

            }
        }
    }

}


std::vector<Edge> MDD::maximize()
{
    std::stack<Node> nodes_to_expand;
    std::vector<Node> expanded;
    Node root = nodes_per_layer[0][0];
    root.set_optimization_value( 0.0 );
    nodes_to_expand.push( root );

    while (nodes_to_expand.size() > 0) {
        Node currentNode = nodes_to_expand.top();
        nodes_to_expand.pop();
        expanded.push_back( currentNode );

        // Many edges can point to the same node
        for (int i = 0; i < currentNode.get_out_edges().size(); ++i) {
            Edge edge = currentNode.get_out_edges()[i];
            Node* next_node = edge.get_head();

            bool node_found = false;
            for (int k = 0; k < expanded.size(); ++k) {
                Node tmp = expanded.at(k);
                if ( &tmp == next_node) {
                    node_found = true;
                    break;
                }
            }

            if ( node_found == false ) {
                // If node does not exists, insert it.
                nodes_to_expand.push( *next_node );
                next_node->set_optimization_value(  currentNode.get_optimization_value() + edge.get_value() );
                currentNode.set_selected_edge( &edge );
            } else {
                // If it does, update its partial solution
                float candidate_value = currentNode.get_optimization_value() + edge.get_value();
                if ( candidate_value > next_node->get_optimization_value() ) {
                    next_node->set_optimization_value( candidate_value );
                    currentNode.set_selected_edge( &edge );
                }

            }

        }

    }

    std::vector<Edge> solution;
    Node current_node = root;
    while (current_node.get_selected_edge() != NULL ) {
        solution.push_back( *current_node.get_selected_edge() );
    }

    return solution;
}

std::vector<Edge> MDD::minimize()
{

}
