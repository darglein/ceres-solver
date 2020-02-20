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
#ifndef CERES_PUBLIC_CODEGEN_INTERNAL_REMOVE_UNUSED_CODE_H_
#define CERES_PUBLIC_CODEGEN_INTERNAL_REMOVE_UNUSED_CODE_H_

#include <iostream>

#include "ceres/codegen/internal/cfg.h"
#include "ceres/codegen/internal/expression_dependencies.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/optimization_pass_summary.h"
namespace ceres {
namespace internal {

inline OptimizationPassSummary RemoveUnusedCode(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "RemoveUnusedCode";

  ExpressionDependencies dep(*graph);

  for (ExpressionId id = graph->Size() - 1; id >= 0; --id) {
    Expression& expr = graph->ExpressionForId(id);
    if (expr.type() == ExpressionType::OUTPUT_ASSIGNMENT) {
      continue;
    }

    if (expr.HasValidLhs() && dep.DataForExpressionId(expr.lhs_id()).Unused()) {
      for (auto e : dep.DataForExpressionId(expr.lhs_id()).written_to) {
        graph->ExpressionForId(e).MakeNop();
        summary.num_expressions_replaced_by_nop++;
      }
      dep = ExpressionDependencies(*graph);
    }
  }

  for (ExpressionId id = graph->Size() - 1; id >= 0; --id) {
    Expression& expr = graph->ExpressionForId(id);
    if (expr.type() == ExpressionType::OUTPUT_ASSIGNMENT ||
        !expr.HasValidLhs()) {
      continue;
    }

    bool found = false;
    for (auto e : dep.DataForExpressionId(expr.lhs_id()).used_by) {
      if (e >= id) {
        found = true;
        break;
      }
    }

    if (!found) {
      graph->ExpressionForId(id).MakeNop();
      summary.num_expressions_replaced_by_nop++;
      dep = ExpressionDependencies(*graph);
    }
  }

  summary.expression_graph_changed =
      summary.num_expressions_replaced_by_nop > 0;
  return summary;
}

inline OptimizationPassSummary ToPartialSSA(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ToPartialSSA";

  ExpressionDependencies expr_dep(*graph);
  CFG cfg(*graph);

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    Expression& expr = graph->ExpressionForId(id);
    auto& dep = expr_dep.DataForExpressionId(id);
    if (!expr.HasValidLhs()) {
      continue;
    }

    if (dep.IsSSA()) {
      continue;
    }

    // find an other assignment in the same BB
    auto b = cfg.BlockIdForExpressionId(id);
    ExpressionId other_id = kInvalidExpressionId;
    for (auto other : dep.written_to) {
      auto other_b = cfg.BlockIdForExpressionId(other);

      if (other_b == b && other > id) {
        other_id = other;
        break;
      }
    }

    if (other_id != kInvalidExpressionId) {
      //      std::cout << "found other expr in the same bb" << std::endl;

      if (expr.lhs_id() == id) {
        // Make other to lhs. Rename everything to other_id expect the
        // expression between them.
        for (auto c : dep.written_to) {
          if (c >= other_id) {
            graph->ExpressionForId(c).UpdateId(id, other_id);
          }
        }
        for (auto c : dep.used_by) {
          if (c >= other_id) {
            graph->ExpressionForId(c).UpdateId(id, other_id);
          }
        }
        summary.num_expressions_modified++;
        expr_dep = ExpressionDependencies(*graph);
      }
      continue;
    }

    //    std::cout << "convert to ssa " << id << std::endl;
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  return summary;
}

inline OptimizationPassSummary TrivialAssignmentElimination(
    ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "TrivialAssignmentElimination";

  ExpressionDependencies deps(*graph);

  for (ExpressionId id = graph->Size() - 1; id >= 0; --id) {
    Expression& expr = graph->ExpressionForId(id);
    auto dep = deps.DataForExpressionId(id);

    if (!expr.HasValidLhs()) {
      continue;
    }

    //    if (expr.type() != ExpressionType::ASSIGNMENT &&
    //        expr.type() != ExpressionType::OUTPUT_ASSIGNMENT) {
    //      continue;
    //    }

    //    std::cout << "found " << id << std::endl;

    for (auto p : expr.arguments()) {
      //      auto p = expr.arguments()[0];

      if (p == expr.lhs_id()) continue;
      auto& other_expr = graph->ExpressionForId(p);

      if (other_expr.type() != ExpressionType::ASSIGNMENT) continue;

      if (!deps.DataForExpressionId(p).IsSSA()) continue;

      auto new_target = other_expr.arguments()[0];
      if (!deps.DataForExpressionId(new_target).IsSSA()) continue;

      expr.UpdateId(p, new_target);
      summary.num_expressions_modified++;
      deps = ExpressionDependencies(*graph);
    }
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  return summary;
}

inline OptimizationPassSummary CombineConstants(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "CombineConstants";

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    Expression& expr = graph->ExpressionForId(id);

    if (expr.type() == ExpressionType::COMPILE_TIME_CONSTANT) {
    }
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  return summary;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_ELIMINATE_NOPS_H_
