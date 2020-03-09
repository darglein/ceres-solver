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

using T = ExpressionRef;

static void checkCFG(const CFG& cfg,
                     const std::vector<CFG::BasicBlock>& nodes) {
  //  cfg.Print();
  EXPECT_EQ(cfg.Size(), nodes.size());

  if (cfg.Size() != nodes.size()) return;

  for (int i = 0; i < cfg.Size(); ++i) {
    auto& node1 = cfg.NodeForId(i);
    auto& node2 = nodes[i];
    EXPECT_EQ(node1.id, node2.id);
    EXPECT_EQ(node1.start, node2.start);
    EXPECT_EQ(node1.end, node2.end);
    EXPECT_EQ(node1.outgoing_edges, node2.outgoing_edges);
  }
}

static void checkDom(const CFG& cfg, const std::vector<BlockId>& doms) {
  EXPECT_EQ(cfg.Size(), doms.size());
  if (cfg.Size() != doms.size()) return;
  for (BlockId i = 0; i < cfg.Size(); ++i) {
    EXPECT_EQ(cfg.Idom(i), doms[i]);
  }
}

TEST(CFG, Size) {
  ExpressionGraph graph;
  EXPECT_EQ(CFG(graph).Size(), 0);
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  EXPECT_EQ(CFG(graph).Size(), 1);
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  EXPECT_EQ(CFG(graph).Size(), 1);
}

TEST(CFG, Blocks) {
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  std::vector<CFG::BasicBlock> reference_nodes = {
      {0, 0, 4, {kEndNode}, {}},
  };
  checkCFG(CFG(graph), reference_nodes);
}

TEST(CFG, If) {
  // CFG
  //
  //  ---------------
  //  |             v
  //  0  ->  1  ->  2  ->  End
  //
  //
  // Dominator Tree
  //
  //       ------ 0 -------
  //       1              2
  //
  StartRecordingExpressions();
  T a = T(0);       // 0
  T b = T(1);       // 1
  auto r1 = a < b;  // 2
  CERES_IF(r1) {    // 3
    a = b;          // 4
  }                 //
  CERES_ENDIF;      // 5
  a = b;            // 6
  auto graph = StopRecordingExpressions();
  std::vector<CFG::BasicBlock> reference_nodes = {
      {0, 0, 4, {1, 2}},
      {1, 4, 6, {2}},
      {2, 6, 7, {kEndNode}},
  };
  std::vector<BlockId> idoms = {0, 0, 0};

  CFG cfg(graph);
  checkCFG(cfg, reference_nodes);
  checkDom(cfg, idoms);
}

TEST(CFG, EmptyIf) {
  // CFG
  //
  //  ---------------
  //  |             v
  //  0  ->  1  ->  2  ->  End
  //
  //
  // Dominator Tree
  //
  //       ------ 0 -------
  //       1              2
  //
  StartRecordingExpressions();
  T a = T(0);       // 0
  T b = T(1);       // 1
  auto r1 = a < b;  // 2
  CERES_IF(r1) {}   // 3
  CERES_ENDIF;      // 5
  a = b;            // 6
  auto graph = StopRecordingExpressions();
  std::vector<CFG::BasicBlock> reference_nodes = {
      {0, 0, 4, {1, 2}, {}},
      {1, 4, 5, {2}, {0}},
      {2, 5, 6, {kEndNode}, {0, 1}},
  };
  std::vector<BlockId> idoms = {0, 0, 0};

  CFG cfg(graph);
  checkCFG(cfg, reference_nodes);
  checkDom(cfg, idoms);
}

TEST(CFG, NestedIf) {
  // CFG
  //
  //         ---------------
  //         |             v
  //  0  ->  1  ->  2  ->  3  ->  4  ->  End
  //  |                           ^
  //  -----------------------------
  //
  // Dominator Tree
  //
  //       ------ 0 -------
  //   ----1----          4
  //   2       3
  //
  StartRecordingExpressions();
  T a = T(0);       // 0
  T b = T(1);       // 1
  auto r1 = a < b;  // 2
  CERES_IF(r1) {    // 3
    a = b;          // 4
    CERES_IF(r1) {  // 5
      a = b;        // 6
    }               //
    CERES_ENDIF;    // 7
    a = b;          // 8
  }                 //
  CERES_ENDIF;      // 9
  a = b;            // 10
  auto graph = StopRecordingExpressions();
  std::vector<CFG::BasicBlock> reference_nodes = {
      {0, 0, 4, {1, 4}},
      {1, 4, 6, {2, 3}},
      {2, 6, 8, {3}},
      {3, 8, 10, {4}},
      {4, 10, 11, {kEndNode}},
  };
  std::vector<BlockId> idoms = {0, 0, 1, 1, 0};

  CFG cfg(graph);
  checkCFG(cfg, reference_nodes);
  checkDom(cfg, idoms);
}

TEST(CFG, EmptyIfNoEnd) {
  // CFG
  //
  //  ----------------
  //  |              v
  //  0  ->  1  ->  End
  //
  //
  // Dominator Tree
  //
  //       ------ 0
  //       1
  //
  StartRecordingExpressions();
  T a = T(0);       // 0
  T b = T(1);       // 1
  auto r1 = a < b;  // 2
  CERES_IF(r1) {}   // 3
  CERES_ENDIF;      // 5
  auto graph = StopRecordingExpressions();
  std::vector<CFG::BasicBlock> reference_nodes = {
      {0, 0, 4, {kEndNode, 1}},
      {1, 4, 5, {kEndNode}},
  };
  std::vector<BlockId> idoms = {0, 0};

  CFG cfg(graph);
  checkCFG(cfg, reference_nodes);
  checkDom(cfg, idoms);
}

TEST(CFG, IfElse) {
  // CFG
  //
  //         ---------------
  //         |             v
  //  0  ->  1      2  ->  3  ->  End
  //  |             ^
  //  ---------------
  //
  // Dominator Tree
  //
  //      ------- 0 --------
  //      1       2        3
  //
  //
  StartRecordingExpressions();
  T a = T(0);       // 0   -   0
  T b = T(1);       // 1   -   0
  auto r1 = a < b;  // 2   -   0
  CERES_IF(r1) {    // 3   -   0
    a = b;          // 4   -   1
  }                 //
  CERES_ELSE {      // 5   -   1
    a = b;          // 6   -   2
  }                 //
  CERES_ENDIF;      // 7   -   2
  a = b;            // 8   -   3
  auto graph = StopRecordingExpressions();
  std::vector<CFG::BasicBlock> reference_nodes = {
      {0, 0, 4, {1, 2}},
      {1, 4, 6, {3}},
      {2, 6, 8, {3}},
      {3, 8, 9, {kEndNode}},
  };
  std::vector<BlockId> idoms = {0, 0, 0, 0};

  CFG cfg(graph);
  checkCFG(cfg, reference_nodes);
  checkDom(cfg, idoms);
}

TEST(CFG, NestedIfElse) {
  // CFG
  //         ---------------      ---------------
  //         |             v      |             v
  //  0  ->  1  ->  2      3  ->  4      5  ->  6  ->  End
  //  |             |             ^      ^
  //  |             ---------------      |
  //  |                                  |
  //  ------------------------------------
  // Dominator Tree
  //
  //      ------- 0 -------------
  // ---- 1 ----       5        6
  // 2    3    4
  //
  StartRecordingExpressions();
  T a = T(0);       // 0   -   0
  T b = T(1);       // 1   -   0
  auto r1 = a < b;  // 2   -   0
  CERES_IF(r1) {    // 3   -   0
    a = b;          // 4   -   1
    CERES_IF(r1) {  // 5   -   1
      a = b;        // 6   -   2
    }               //
    CERES_ELSE {    // 7   -   2
      a = b;        // 8   -   3
    }               //
    CERES_ENDIF;    // 9   -   3
    a = b;          // 10  -   4
  }                 //
  CERES_ELSE {      // 11  -   4
    a = b;          // 12  -   5
  }                 //
  CERES_ENDIF;      // 13  -   5
  a = b;            // 14  -   6
  auto graph = StopRecordingExpressions();
  std::vector<CFG::BasicBlock> reference_nodes = {
      {0, 0, 4, {1, 5}},
      {1, 4, 6, {2, 3}},
      {2, 6, 8, {4}},
      {3, 8, 10, {4}},
      {4, 10, 12, {6}},
      {5, 12, 14, {6}},
      {6, 14, 15, {kEndNode}},
  };
  std::vector<BlockId> idoms = {0, 0, 1, 1, 1, 0, 0};

  CFG cfg(graph);
  checkCFG(cfg, reference_nodes);
  checkDom(cfg, idoms);
}

}  // namespace internal
}  // namespace ceres
