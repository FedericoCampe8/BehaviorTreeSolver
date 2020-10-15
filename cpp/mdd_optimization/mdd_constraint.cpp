#include "mdd_optimization/mdd_constraint.hpp"

#include <stdexcept>  // for std::invalid_argument

namespace mdd {

MDDConstraint::MDDConstraint(ConstraintType type, const std::string& name)
: Constraint(type, name)
{
}


void MDDConstraint::eraseUnfeasibleSuccessors(Node* node, Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const
{
    std::vector<Node*> nodesToDelete;
    std::queue<Node*> queue;
    queue.push( node );

    while (queue.size() > 0) {
      Node* currNode = queue.front();
      queue.pop();
      std::vector<Node*> children;
      // Get all unique children
      for (auto outEdge: currNode->getOutEdges()) {
          if ( std::count(children.begin(),children.end(),outEdge->getHead()) == 0 ) {
              children.push_back( outEdge->getHead() );
          }
      }

      // Iterate through children
      for (auto nextNode : children) {
          bool nextNodeValid = false;
          for (auto headInEdge: nextNode->getInEdges()) {
              if (headInEdge->getTail() != currNode) {
                nextNodeValid = true;
              }
          }

          if (nextNodeValid == false) {
              queue.push( nextNode );
              nodesToDelete.push_back( nextNode );
          } 
      }

      // At this point curreNode is no longer valid.
      arena->deleteNode(currNode->getUniqueId());
      for (auto edge: currNode->getOutEdges()) {
          arena->deleteEdge( edge->getUniqueId() );
          edge->removeEdgeFromNodes();
      }

      mddRepresentation[currNode->getLayer()].erase( std::find(mddRepresentation[currNode->getLayer()].begin(), 
              mddRepresentation[currNode->getLayer()].end(), currNode) );


    }
}

void MDDConstraint::eraseUnfeasiblePredecessors(Node* node, Arena* arena, std::vector<std::vector<Node*>>& mddRepresentation) const
{
    std::vector<Node*> nodesToDelete;
    std::queue<Node*> queue;
    queue.push( node );

    while (queue.size() > 0) {
      Node* currNode = queue.front();
      queue.pop();
      std::vector<Node*> parents;
      // Get all unique children
      for (auto inEdge: currNode->getInEdges()) {
          if ( std::count(parents.begin(),parents.end(), inEdge->getTail()) == 0 ) {
              parents.push_back( inEdge->getTail() );
          }
      }

      // Iterate through children
      for (auto prevNode : parents) {
          bool prevNodeValid = false;
          for (auto prevOutEdge: prevNode->getOutEdges()) {
              if (prevOutEdge->getHead() != currNode) {
                prevNodeValid = true;
              }
          }

          if (prevNodeValid == false) {
              queue.push( prevNode );
              nodesToDelete.push_back( prevNode );
          } 
      }

      // At this point curreNode is no longer valid.
      arena->deleteNode(currNode->getUniqueId());
      for (auto edge: currNode->getInEdges()) {
          arena->deleteEdge( edge->getUniqueId() );
          edge->removeEdgeFromNodes();
      }

      mddRepresentation[currNode->getLayer()].erase( std::find(mddRepresentation[currNode->getLayer()].begin(), 
              mddRepresentation[currNode->getLayer()].end(), currNode) );

    }

}


}  // namespace mdd

