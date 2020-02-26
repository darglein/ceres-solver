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
#include "ceres/codegen/internal/remove_common_subexpressions.h"

#include <unordered_map>

namespace ceres {
namespace internal {

static size_t RhsExpressionHash(const Expression& expr) {
  constexpr size_t max_code_size = 1000 * 1000;
  size_t result = 0;
  result += static_cast<size_t>(expr.type()) + max_code_size;
  size_t i = 2;
  for (auto a : expr.arguments()) {
    result += a + max_code_size * i;
    i++;
  }
  return result;
}

OptimizationPassSummary RemoveCommonSubexpressions(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "RemoveCommonSubexpressions";
  summary.start();
  ExpressionDependencies dep(*graph);
  CFG cfg(*graph);

  std::unordered_multimap<size_t, ExpressionId> map;
  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    auto& expr = graph->ExpressionForId(id);
    if (!expr.HasValidLhs()) continue;
    if (!dep.DataForExpressionId(expr.lhs_id()).IsSSA()) continue;
    map.insert({RhsExpressionHash(expr), id});
  }

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    auto& expr = graph->ExpressionForId(id);
    if (!expr.HasValidLhs()) continue;
    if (!dep.DataForExpressionId(expr.lhs_id()).IsSSA()) continue;

    bool all_params_ssa = true;
    for (auto& p : expr.arguments()) {
      CHECK(p != kInvalidExpressionId);
      if (!dep.DataForExpressionId(p).IsSSA()) {
        all_params_ssa = false;
        break;
      }
    }

    if (!all_params_ssa) continue;

    //    map.find()

    //    continue;
    //    for (ExpressionId other = 0; other < graph->Size(); ++other) {
    auto its = map.equal_range(RhsExpressionHash(expr));

    for (auto it = its.first; it != its.second; ++it) {
      auto other = it->second;
      auto& other_expr = graph->ExpressionForId(other);
      if (other == id) continue;
      if (!other_expr.HasValidLhs()) continue;
      if (!dep.DataForExpressionId(other_expr.lhs_id()).IsSSA()) continue;

      if (expr.IsReplaceableBy(other_expr) &&
          cfg.DominateExpression(id, other)) {
        //        std::cout << "replaceable " << id << " " << other <<
        //        std::endl;
        // replace by assignment
        other_expr.Replace(
            Expression::CreateAssignment(kInvalidExpressionId, id));
        summary.num_expressions_modified++;
        dep.rebuild();
      }
    }
  }

  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  summary.end();
  return summary;
}

}  // namespace internal
}  // namespace ceres
