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

#include "ceres/codegen/internal/optimize_expression_graph.h"

#include "ceres/codegen/internal/constant_optimization.h"
#include "ceres/codegen/internal/eliminate_nops.h"
#include "ceres/codegen/internal/merge_constants.h"
#include "ceres/codegen/internal/remove_common_subexpressions.h"
#include "ceres/codegen/internal/remove_unused_code.h"
#include "glog/logging.h"
namespace ceres {
namespace internal {

std::ostream& operator<<(std::ostream& strm,
                         const OptimizeExpressionGraphSummary& summary) {
  strm << "ExpressionGraph Optimization Summary" << std::endl;
  strm << "Num Iterations: " << summary.num_iterations << std::endl;
  for (auto& pass_summary : summary.summaries) {
    strm << pass_summary << std::endl;
  }
  return strm;
}

OptimizeExpressionGraphSummary OptimizeExpressionGraph(
    const OptimizeExpressionGraphOptions& options, ExpressionGraph* graph) {
  OptimizeExpressionGraphSummary summary;
  summary.num_iterations = 0;
  while (summary.num_iterations < options.max_num_iterations) {
    summary.num_iterations++;
    bool changed = false;
    {
      auto pass_summary = MergeConstants(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }
    if (options.eliminate_nops) {
      auto pass_summary = EliminateNops(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }

    {
      auto pass_summary = RemoveUnusedCode(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }

    {
      auto pass_summary = ToPartialSSA(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }

    {
      auto pass_summary = TrivialAssignmentElimination(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }
    {
      auto pass_summary = ConstantFolding(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }

    {
      auto pass_summary = ForwardFlow(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }
#if 1

    {
      auto pass_summary = ZeroOnePropagation(graph, false);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }

    if (options.eliminate_nops) {
      auto pass_summary = EliminateNops(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }

    {
      auto pass_summary = SortArguments(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }

    //    if (summary.num_iterations > 2)
    if (1) {
      auto pass_summary = RemoveCommonSubexpressions(graph);
      changed |= pass_summary.expression_graph_changed;
      summary.summaries.push_back(pass_summary);
    }
#endif

#if 0
#endif
    if (!changed) {
      break;
    }
  }

  return summary;
}

OptimizeExpressionGraphSummary SuperOptimize(
    const OptimizeExpressionGraphOptions& options, ExpressionGraph* graph) {
  //  return OptimizeExpressionGraph(options, graph);
  auto a1 = OptimizeExpressionGraph(options, graph);
  Reorder(graph, true, "*");
  Reorder(graph, true, "+");
  MoveToUsage(graph);
  auto a2 = OptimizeExpressionGraph(options, graph);
  Reorder(graph, false, "*");
  Reorder(graph, false, "+");
  MoveToUsage(graph);
  auto a3 = OptimizeExpressionGraph(options, graph);
  return a3;
}

}  // namespace internal
}  // namespace ceres
