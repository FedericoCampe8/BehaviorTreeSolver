#include <gtest/gtest.h>

#include <iostream>
#include <limits>   // for std::numeric_limits

#include "bt/behavior_tree_arena.hpp"
#include "bt/edge.hpp"
#include "bt/node.hpp"

TEST(TestNode, SetEdges)
{
  auto arena = std::make_unique<btsolver::BehaviorTreeArena>();
  auto n1 = std::make_unique<btsolver::Node>("n1", btsolver::NodeType::UndefinedType, arena.get());

  auto e1 = std::make_unique<btsolver::Edge>();
  auto e2 = std::make_unique<btsolver::Edge>();
  auto e3 = std::make_unique<btsolver::Edge>();
  auto e4 = std::make_unique<btsolver::Edge>();

  EXPECT_TRUE(n1->getAllIncomingEdges().empty());
  EXPECT_TRUE(n1->getAllOutgoingEdges().empty());

  EXPECT_NO_THROW(n1->addIncomingEdge(e1.get()));
  EXPECT_NO_THROW(n1->addIncomingEdge(e2.get()));
  EXPECT_NO_THROW(n1->addOutgoingEdge(e3.get()));
  EXPECT_NO_THROW(n1->addOutgoingEdge(e4.get()));

  EXPECT_EQ(n1->getAllIncomingEdges().size(), 2);
  EXPECT_EQ(n1->getAllOutgoingEdges().size(), 2);

  EXPECT_EQ(e1->getTail(), n1.get());
  EXPECT_EQ(e2->getTail(), n1.get());
  EXPECT_EQ(e3->getHead(), n1.get());
  EXPECT_EQ(e4->getHead(), n1.get());

  EXPECT_NO_THROW(n1->removeIncomingEdge(e1.get()));
  EXPECT_NO_THROW(n1->removeOutgoingEdge(e3.get()));

  EXPECT_EQ(n1->getAllIncomingEdges().size(), 1);
  EXPECT_EQ(n1->getAllOutgoingEdges().size(), 1);

  EXPECT_EQ(e1->getTail(), nullptr);
  EXPECT_EQ(e2->getTail(), n1.get());
  EXPECT_EQ(e3->getHead(), nullptr);
  EXPECT_EQ(e4->getHead(), n1.get());
}
