#include <gtest/gtest.h>

#include <iostream>
#include <limits>   // for std::numeric_limits

#include "bt/behavior_tree_arena.hpp"
#include "bt/edge.hpp"
#include "bt/node.hpp"

TEST(TestEdge, DefaultConstructor)
{
  btsolver::Edge edge;
  EXPECT_EQ(edge.getDomainLowerBound(), std::numeric_limits<int32_t>::min());
  EXPECT_EQ(edge.getDomainUpperBound(), std::numeric_limits<int32_t>::max());
}

TEST(TestEdge, SetDomainBounds)
{
  btsolver::Edge edge;

  const int32_t lb{1};
  const int32_t ub{3};
  EXPECT_NO_THROW(edge.setDomainBounds(lb, ub));
  EXPECT_TRUE(edge.isParallelEdge());

  ASSERT_EQ(lb, edge.getDomainLowerBound());
  ASSERT_EQ(ub, edge.getDomainUpperBound());

  for (int32_t d{lb}; d <= ub; ++d)
  {
    EXPECT_TRUE(edge.isElementInDomain(d));
  }
  EXPECT_FALSE(edge.isDomainEmpty());
  EXPECT_EQ(3, edge.getDomainSize());
}

TEST(TestEdge, OperateOnDomain)
{
  btsolver::Edge edge;

  const int32_t lb{1};
  const int32_t ub{3};
  EXPECT_NO_THROW(edge.setDomainBounds(lb, ub));

  uint32_t ctr{3};
  for (int32_t d{lb}; d <= ub; ++d)
  {
    EXPECT_EQ(ctr--, edge.getDomainSize());
    EXPECT_NO_THROW(edge.removeElementFromDomain(d));
    EXPECT_FALSE(edge.isElementInDomain(d));
  }
  EXPECT_TRUE(edge.isDomainEmpty());
  EXPECT_EQ(0, edge.getDomainSize());

  for (int32_t d{ub}; d >= lb; --d)
  {
    EXPECT_NO_THROW(edge.reinsertElementInDomain(d));
    EXPECT_TRUE(edge.isElementInDomain(d));
  }
  EXPECT_FALSE(edge.isDomainEmpty());
  ASSERT_EQ(lb, edge.getDomainLowerBound());
  ASSERT_EQ(ub, edge.getDomainUpperBound());
}

TEST(TestEdge, SetNodes)
{
  auto arena = std::make_unique<btsolver::BehaviorTreeArena>();
  auto n1 = std::make_unique<btsolver::Node>("n1", btsolver::NodeType::UndefinedType, arena.get());
  auto n2 = std::make_unique<btsolver::Node>("n2", btsolver::NodeType::UndefinedType, arena.get());
  auto n3 = std::make_unique<btsolver::Node>("n3", btsolver::NodeType::UndefinedType, arena.get());

  btsolver::Edge edge;
  EXPECT_NO_THROW(edge.removeHead());
  EXPECT_NO_THROW(edge.removeTail());
  EXPECT_TRUE(edge.getHead() == nullptr);
  EXPECT_TRUE(edge.getTail() == nullptr);

  EXPECT_TRUE(n1->getAllIncomingEdges().empty());
  EXPECT_TRUE(n2->getAllIncomingEdges().empty());
  EXPECT_TRUE(n3->getAllIncomingEdges().empty());

  EXPECT_TRUE(n1->getAllOutgoingEdges().empty());
  EXPECT_TRUE(n2->getAllOutgoingEdges().empty());
  EXPECT_TRUE(n3->getAllOutgoingEdges().empty());

  EXPECT_NO_THROW(edge.setHead(n1.get()));
  EXPECT_NO_THROW(edge.setTail(n2.get()));
  EXPECT_TRUE(edge.getHead() == n1.get());
  EXPECT_TRUE(edge.getTail() == n2.get());

  EXPECT_EQ(n1->getAllOutgoingEdges().size(), 1);
  EXPECT_EQ(n2->getAllIncomingEdges().size(), 1);

  // Set a new tail
  EXPECT_NO_THROW(edge.setTail(n3.get()));
  EXPECT_TRUE(edge.getTail() == n3.get());
  EXPECT_EQ(n2->getAllIncomingEdges().size(), 0);
  EXPECT_EQ(n3->getAllIncomingEdges().size(), 1);

  EXPECT_NO_THROW(edge.removeEdgeFromNodes());
  EXPECT_TRUE(edge.getHead() == nullptr);
  EXPECT_TRUE(edge.getTail() == nullptr);

  EXPECT_TRUE(n1->getAllIncomingEdges().empty());
  EXPECT_TRUE(n2->getAllIncomingEdges().empty());
  EXPECT_TRUE(n3->getAllIncomingEdges().empty());

  EXPECT_TRUE(n1->getAllOutgoingEdges().empty());
  EXPECT_TRUE(n2->getAllOutgoingEdges().empty());
  EXPECT_TRUE(n3->getAllOutgoingEdges().empty());
}
