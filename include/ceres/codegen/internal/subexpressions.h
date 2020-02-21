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
#ifndef CERES_PUBLIC_CODEGEN_INTERNAL_SUBEXPRESSIONS_H_
#define CERES_PUBLIC_CODEGEN_INTERNAL_SUBEXPRESSIONS_H_

#include <iostream>

#include "ceres/codegen/internal/cfg.h"
#include "ceres/codegen/internal/expression_dependencies.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/optimization_pass_summary.h"
namespace ceres {
namespace internal {

inline OptimizationPassSummary RemoveCommonSubexpressions(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "RemoveCommonSubexpressions";

  ExpressionDependencies dep(*graph);
  CFG cfg(*graph);

  for(ExpressionId id = 0; id < graph->Size(); ++id)
  {
    auto& expr = graph->ExpressionForId(id);
    if(!dep.DataForExpressionId(id).IsSSA()) continue;
    if(!expr.HasValidLhs()) continue;

    bool all_params_ssa = true;
    for(auto& p : expr.arguments())
    {
      if(!dep.DataForExpressionId(p).IsSSA()){
        all_params_ssa = false;
        break;
      }
    }

    if(!all_params_ssa)
      continue;

    for(ExpressionId other = 0; other < graph->Size(); ++other)
    {
      auto& other_expr = graph->ExpressionForId(other);
      if(other == id) continue;
      if(!dep.DataForExpressionId(other).IsSSA()) continue;
      if(!expr.HasValidLhs()) continue;

      if(expr.IsReplaceableBy(other_expr) && cfg.DominateExpression(id,other))
      {
//        std::cout << "replaceable " << id << " " << other << std::endl;
        // replace by assignment
        other_expr.Replace(Expression::CreateAssignment(kInvalidExpressionId,id));
        summary.num_expressions_modified++;
      }
    }

  }


  summary.expression_graph_changed =
      summary.num_expressions_modified > 0;
  return summary;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_ELIMINATE_NOPS_H_
