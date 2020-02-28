// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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
#include "autodiff_codegen_test.h"
#include "ceres/codegen/generate_code_for_functor.h"
#include "ceres/codegen/internal/code_generator.h"
#include "ceres/codegen/internal/constant_optimization.h"
#include "ceres/codegen/internal/eliminate_nops.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/expression_ref.h"
#include "ceres/codegen/internal/merge_constants.h"
#include "ceres/codegen/internal/optimize_expression_graph.h"
#include "ceres/codegen/internal/remove_common_subexpressions.h"
#include "ceres/codegen/internal/remove_unused_code.h"
#include "ceres/jet.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using T = ExpressionRef;

static void GenerateAndCheck(const ExpressionGraph& graph) {
  CodeGenerator::Options generator_options;
  CodeGenerator gen(graph, generator_options);
  auto code = gen.Generate();

  for (int i = 0; i < code.size(); ++i) {
    std::cout << code[i] << std::endl;
  }
}
#if 0
TEST(ToPartialSSA, SimpleLinear) {
  StartRecordingExpressions();
  {
    T a = T(0);
    T b = T(2);
    a = b;
    T c = a;
  }
  auto graph = StopRecordingExpressions();

  GenerateAndCheck(graph);
  bool changed = true;
  while (changed) {
    auto summary = ToPartialSSA(&graph);
    changed = summary.expression_graph_changed;
    std::cout << summary << std::endl;
  }

  GenerateAndCheck(graph);
}
#endif

TEST(ToPartialSSA, Flow) {
  StartRecordingExpressions();
  using T = ExpressionRef;
  {
    T c1 = T(1);
    T c2 = T(2);

    T v = T(0);
    CERES_IF(c1 < c2) { v = T(1); }
    CERES_ELSE { v = T(1); }
    CERES_ENDIF;
    //    T c3 = c1;
    //        c3 = v;
    T c3 = v;
    c3 = v;
    T c4 = c3;

    auto y = c4;
    MakeOutput(y, "a");
  }
  auto graph = StopRecordingExpressions();

  GenerateAndCheck(graph);
  OptimizeExpressionGraph(OptimizeExpressionGraphOptions(), &graph);
  GenerateAndCheck(graph);
}

TEST(ToPartialSSA, If) {
  return;
  //  auto res =
  //      GenerateCodeForFunctor<test::ScalarFunctions>(AutoDiffCodeGenOptions());

  //  return;
  StartRecordingExpressions();
  using T = Jet<ExpressionRef, 1>;
  //  using T = ExpressionRef;
  {
    //    T x = T(MakeParameter("input"));
    //    T a = sin(x);
    //    T a2 = sin(x);

    T c1 = T(1);
    T c2 = T(2);
    T c3 = T(3);
    //    T c4 = T(4);
    //    T c5 = T(5);
    //    T c6 = T(6);

    T v1 = c3 * c1;
    T v0 = atan2(v1, c1);
    T v2 = v1 * v0;

    //    T v3 = c1 * c3;
    //    T v4 = v3 * c2;
    //    MakeOutput(v4, "out");
    //    T b = a + a2;
    //    T b = pow(a, a2);

    //    T y = v2 + v4;
    //    MakeOutput(v2, "a");
    auto y = v2;
    MakeOutput(y.a, "a");
    MakeOutput(y.v[0], "a");
    //    MakeOutput(b.v[1], "a");
  }
  auto graph = StopRecordingExpressions();

  GenerateAndCheck(graph);
  bool changed = true;
  while (changed) {
    changed = false;
#if 1

    {
      auto summary = EliminateNops(&graph);
      changed |= summary.expression_graph_changed;
    }

    {
      auto summary = RemoveUnusedCode(&graph);
      changed |= summary.expression_graph_changed;
    }
    {
      auto summary = ToPartialSSA(&graph);
      changed |= summary.expression_graph_changed;
    }
    {
      auto summary = TrivialAssignmentElimination(&graph);
      changed |= summary.expression_graph_changed;
    }
    {
      auto summary = MergeConstants(&graph);
      changed |= summary.expression_graph_changed;
    }
    {
      auto summary = TrivialAssignmentElimination(&graph);
      changed |= summary.expression_graph_changed;
    }

    //    {
    //      auto summary = ZeroOnePropagation(&graph);
    //      changed |= summary.expression_graph_changed;
    //    }
    {
      auto summary = RemoveCommonSubexpressions(&graph);
      changed |= summary.expression_graph_changed;
      std::cout << summary << std::endl;
    }
#endif
    std::cout << std::endl << std::endl << std::endl;
    GenerateAndCheck(graph);
    {
      auto summary = Reorder(&graph, false, "*");
      changed |= summary.expression_graph_changed;
      std::cout << summary << std::endl;
    }

    GenerateAndCheck(graph);
    {
      auto summary = MoveToUsage(&graph);
      changed |= summary.expression_graph_changed;
      std::cout << summary << std::endl;
    }
    GenerateAndCheck(graph);
    //    {
    //      auto summary = MoveToUsage(&graph);
    //      changed |= summary.expression_graph_changed;
    //    }
  }

  GenerateAndCheck(graph);
}

}  // namespace internal
}  // namespace ceres
