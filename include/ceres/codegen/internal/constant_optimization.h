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
#ifndef CERES_PUBLIC_CODEGEN_INTERNAL_CONSTANT_OPTIMIZATION_H_
#define CERES_PUBLIC_CODEGEN_INTERNAL_CONSTANT_OPTIMIZATION_H_

#include <cmath>
#include <iostream>
#include <limits>

#include "ceres/codegen/internal/cfg.h"
#include "ceres/codegen/internal/expression_dependencies.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/optimization_pass_summary.h"
namespace ceres {
namespace internal {

// Two passes.
//   1. Move all compile time constants to the beginning.
inline OptimizationPassSummary ReorderCompileTimeConstants(
    ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ReorderCompileTimeConstants";

  ExpressionId end_of_constant_section = 0;
  for (; end_of_constant_section < graph->Size(); ++end_of_constant_section) {
    if (graph->ExpressionForId(end_of_constant_section).type() !=
        ExpressionType::COMPILE_TIME_CONSTANT)
      break;
  }
  if (end_of_constant_section == graph->Size()) return summary;

  for (ExpressionId id = end_of_constant_section; id < graph->Size(); ++id) {
    auto& expr = graph->ExpressionForId(id);
    if (expr.type() == ExpressionType::COMPILE_TIME_CONSTANT) {
      // We found a constant, which is not in the constant section.

      // Insert a constant expression at the end of the section.
      Expression cpy = expr;
      cpy.set_lhs_id(end_of_constant_section);
      graph->Insert(end_of_constant_section, cpy);

      // replace by nop
      graph->ExpressionForId(id + 1).MakeNop();

      // Update
      for (ExpressionId later_id = id + 1; later_id < graph->Size();
           ++later_id) {
        graph->ExpressionForId(later_id).UpdateId(id + 1,
                                                  end_of_constant_section);
      }
      summary.num_expressions_inserted++;
      summary.num_expressions_replaced_by_nop++;
      summary.expression_graph_changed = true;
    }
  }
  return summary;
}

inline bool finiteDoubleEquality(double v1, double v2) {
  // If both are inf or nan it's fine too!
  if (std::isinf(v1) && std::isinf(v2)) {
    return std::signbit(v1) == std::signbit(v2);
  }

  if (std::isnan(v1) && std::isnan(v2)) {
    return true;
  }

  return v1 == v2;
}
// Merge compile time constants of the same value.
// Only merges values from the constant section
inline OptimizationPassSummary MergeCompileTimeConstants(
    ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "MergeCompileTimeConstants";

  ExpressionId end_of_constant_section = 0;
  for (; end_of_constant_section < graph->Size(); ++end_of_constant_section) {
    if (graph->ExpressionForId(end_of_constant_section).type() !=
        ExpressionType::COMPILE_TIME_CONSTANT)
      break;
  }
  if (end_of_constant_section == graph->Size()) return summary;

  for (ExpressionId id = 0; id < end_of_constant_section; ++id) {
    auto& expr = graph->ExpressionForId(id);

    // this check is required because multipe constants can be merged in one
    // pass
    if (expr.type() != ExpressionType::COMPILE_TIME_CONSTANT) continue;

    // find a constant value
    for (ExpressionId other_id = 0; other_id < id; ++other_id) {
      auto& other_expr = graph->ExpressionForId(other_id);
      // this check is required because multipe constants can be merged in one
      // pass
      if (other_expr.type() != ExpressionType::COMPILE_TIME_CONSTANT) continue;

      if (finiteDoubleEquality(expr.value(), other_expr.value())) {
        // replace expr with other expr
        expr.MakeNop();
        summary.num_expressions_replaced_by_nop++;
        summary.expression_graph_changed = true;
        for (ExpressionId i = 0; i < graph->Size(); ++i) {
          graph->ExpressionForId(i).UpdateId(id, other_id);
        }

        break;
      }
    }
  }

  return summary;
}

// Merge compile time constants of the same value.
// Only merges values from the constant section
inline OptimizationPassSummary ZeroOnePropagation(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ZeroOnePropagation";

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    auto& expr = graph->ExpressionForId(id);

    if (expr.type() == ExpressionType::BINARY_ARITHMETIC) {
      if (expr.name() == "*") {
        if (graph->ExpressionForId(expr.arguments()[0])
                .IsCompileTimeConstantAndEqualTo(0) ||
            graph->ExpressionForId(expr.arguments()[1])
                .IsCompileTimeConstantAndEqualTo(0)) {
          // One of the operants is 0
          // -> replace with constant 0
          expr.Replace(Expression::CreateCompileTimeConstant(0));
          summary.num_expressions_modified++;
          continue;
        }
        if (graph->ExpressionForId(expr.arguments()[0])
                .IsCompileTimeConstantAndEqualTo(1)) {
          expr.Replace(Expression::CreateAssignment(kInvalidExpressionId,
                                                    expr.arguments()[1]));
          summary.num_expressions_modified++;
          continue;
        }
        if (graph->ExpressionForId(expr.arguments()[1])
                .IsCompileTimeConstantAndEqualTo(1)) {
          expr.Replace(Expression::CreateAssignment(kInvalidExpressionId,
                                                    expr.arguments()[0]));
          summary.num_expressions_modified++;
          continue;
        }
      }
      if (expr.name() == "/") {
        if (graph->ExpressionForId(expr.arguments()[0])
                .IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateCompileTimeConstant(0));
          summary.num_expressions_modified++;
          continue;
        }
        if (graph->ExpressionForId(expr.arguments()[1])
                .IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateCompileTimeConstant(1.0 / 0.0));
          summary.num_expressions_modified++;
          continue;
        }
      }
      if (expr.name() == "+") {
        if (graph->ExpressionForId(expr.arguments()[0])
                .IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateAssignment(kInvalidExpressionId,
                                                    expr.arguments()[1]));
          summary.num_expressions_modified++;
          continue;
        }
        if (graph->ExpressionForId(expr.arguments()[1])
                .IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateAssignment(kInvalidExpressionId,
                                                    expr.arguments()[0]));
          summary.num_expressions_modified++;
          continue;
        }
      }
      if (expr.name() == "-") {
        if (graph->ExpressionForId(expr.arguments()[0])
                .IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(
              Expression::CreateUnaryArithmetic("-", expr.arguments()[1]));
          summary.num_expressions_modified++;
          continue;
        }
        if (graph->ExpressionForId(expr.arguments()[1])
                .IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateAssignment(kInvalidExpressionId,
                                                    expr.arguments()[0]));
          summary.num_expressions_modified++;
          continue;
        }
      }
    }
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  return summary;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_ELIMINATE_NOPS_H_
