// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
#define CERES_CODEGEN

#include "ceres/codegen/internal/cfg.h"

#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/expression_ref.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

static void checkCFG(const CFG& cfg, const std::vector<CFG::Node>& nodes) {
  EXPECT_EQ(cfg.Size(), nodes.size());

  if (cfg.Size() != nodes.size()) return;

  for (int i = 0; i < cfg.Size(); ++i) {
    auto& node1 = cfg.NodeForId(i);
    auto& node2 = nodes[i];
    EXPECT_EQ(node1.id, node2.id);
    EXPECT_EQ(node1.outgoing_edges, node2.outgoing_edges);
    EXPECT_EQ(node1.incoming_edges, node2.incoming_edges);
  }
}

TEST(CFG, Size) {
  ExpressionGraph graph;
  EXPECT_EQ(CFG(graph).Size(), 0);
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  EXPECT_EQ(CFG(graph).Size(), 1);
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  EXPECT_EQ(CFG(graph).Size(), 3);
}

TEST(CFG, Linear) {
  // A linear graph of the following structure
  // 0 -> 1 -> 2 -> 3 -> end
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  std::vector<CFG::Node> reference_nodes = {
      {0, {1}, {}},
      {1, {2}, {0}},
      {2, {3}, {1}},
      {3, {-1}, {2}},
  };
  checkCFG(CFG(graph), reference_nodes);
}

TEST(CFG, If) {
  //
  //  0 -> 1 -> 2 -> 3 (if) -> 4 -> 5 (endif) -> 6
  //                 |                           ^
  //                 ----------------------------
  //
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(1));
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  graph.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  std::vector<CFG::Node> reference_nodes = {
      {0, {1}, {}},
      {1, {2}, {0}},
      {2, {3}, {1}},
      {3, {4, 6}, {2}},
      {4, {5}, {3}},
      {5, {6}, {4}},
      {6, {-1}, {3, 5}},
  };
  checkCFG(CFG(graph), reference_nodes);
}

TEST(CFG, IfElse) {
  //                                -------------------------------
  //                                |                             v
  //  0 -> 1 -> 2 -> 3 (if) -> 4 -> 5 (else)    6 -> 7 (endif) -> 8
  //                 |                          ^
  //                 ----------------------------
  //
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(1));
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  graph.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  graph.InsertBack(Expression::CreateElse());
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  std::vector<CFG::Node> reference_nodes = {
      {0, {1}, {}},
      {1, {2}, {0}},
      {2, {3}, {1}},
      {3, {4, 6}, {2}},
      {4, {5}, {3}},
      {5, {8}, {4}},
      {6, {7}, {3}},
      {7, {8}, {6}},
      {8, {-1}, {5, 7}},
  };
  checkCFG(CFG(graph), reference_nodes);
}

}  // namespace internal
}  // namespace ceres
