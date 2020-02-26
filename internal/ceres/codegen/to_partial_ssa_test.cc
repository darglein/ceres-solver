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

#include "ceres/codegen/internal/code_generator.h"
#include "ceres/codegen/internal/constant_optimization.h"
#include "ceres/codegen/internal/eliminate_nops.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/expression_ref.h"
#include "ceres/codegen/internal/optimize_expression_graph.h"
#include "ceres/codegen/internal/remove_unused_code.h"
#include "ceres/codegen/internal/subexpressions.h"
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

TEST(ToPartialSSA, If) {
  StartRecordingExpressions();
  using T = Jet<ExpressionRef, 2>;
  {
    T x = T(MakeParameter("input"));
    T a = sin(x);
    T a2 = sin(x);

    T b = a + a2;
    //    T b = pow(a, a2);

    MakeOutput(b.a, "a");
    MakeOutput(b.v[0], "a");
    MakeOutput(b.v[1], "a");
  }
  auto graph = StopRecordingExpressions();

  GenerateAndCheck(graph);
  bool changed = true;
  while (changed) {
    changed = false;

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
      auto summary = MoveConstantsToBeginning(&graph);
      changed |= summary.expression_graph_changed;
    }
    {
      auto summary = MergeCompileTimeConstants(&graph);
      changed |= summary.expression_graph_changed;
    }
    {
      auto summary = ZeroOnePropagation(&graph);
      changed |= summary.expression_graph_changed;
    }
    {
      auto summary = RemoveCommonSubexpressions(&graph);
      changed |= summary.expression_graph_changed;
    }
  }

  GenerateAndCheck(graph);
}

}  // namespace internal
}  // namespace ceres
