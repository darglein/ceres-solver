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

// Applies the following optimizations:
//
//  Zero Propagation
//    v_1 = 0 * v_0;        ->        v_1 = 0
//    v_1 = v_0 * 0;        ->        v_1 = 0
//    v_1 = 0 / v_0;        ->        v_1 = 0
//    v_1 = v_0 / 0;        ->        v_1 = nan
//    v_1 = 0 + v_0;        ->        v_1 = v_0
//    v_1 = v_0 + 0;        ->        v_1 = v_0
//    v_1 = 0 - v_0;        ->        v_1 = -v_0
//    v_1 = v_0 - 0;        ->        v_1 = v_0
//  One Propagation
//    v_1  = 1 * v_0;       ->        v_1 = v_0
//    v_1  = v_0 * 1;       ->        v_1 = v_0
//    v_1  = v_0 / 1;       ->        v_1 = v_0
//
// In strict-mode, transformation that don't conform with the IEEE 754 floating
// point rules are not performed. This affects the multiplication and division
// by zero.
inline OptimizationPassSummary ZeroOnePropagation(
    ExpressionGraph* graph, bool strict_ieee_float = false) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ZeroOnePropagation";
  summary.start();

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    auto& expr = graph->ExpressionForId(id);

    if (expr.type() == ExpressionType::BINARY_ARITHMETIC) {
      auto left_id = expr.arguments()[0];
      auto right_id = expr.arguments()[1];
      auto& left_expr = graph->ExpressionForId(left_id);
      auto& right_expr = graph->ExpressionForId(right_id);

      if (expr.name() == "*") {
        if (!strict_ieee_float &&
            (left_expr.IsCompileTimeConstantAndEqualTo(0) ||
             right_expr.IsCompileTimeConstantAndEqualTo(0))) {
          // One of the operants is 0
          // -> replace with constant 0
          expr.Replace(Expression::CreateCompileTimeConstant(0));
          summary.num_expressions_modified++;
        } else if (left_expr.IsCompileTimeConstantAndEqualTo(1)) {
          expr.Replace(
              Expression::CreateAssignment(kInvalidExpressionId, right_id));
          summary.num_expressions_modified++;
        } else if (right_expr.IsCompileTimeConstantAndEqualTo(1)) {
          expr.Replace(
              Expression::CreateAssignment(kInvalidExpressionId, left_id));
          summary.num_expressions_modified++;
        } else if (left_expr.IsCompileTimeConstantAndEqualTo(-1)) {
          expr.Replace(Expression::CreateUnaryArithmetic("-", right_id));
          summary.num_expressions_modified++;
        } else if (right_expr.IsCompileTimeConstantAndEqualTo(-1)) {
          expr.Replace(Expression::CreateUnaryArithmetic("-", left_id));
          summary.num_expressions_modified++;
        }
      } else if (expr.name() == "/") {
        if (!strict_ieee_float &&
            left_expr.IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateCompileTimeConstant(0));
          summary.num_expressions_modified++;
        } else if (!strict_ieee_float &&
                   right_expr.IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateCompileTimeConstant(
              std::numeric_limits<double>::quiet_NaN()));
          summary.num_expressions_modified++;
        }
      } else if (expr.name() == "+") {
        if (left_expr.IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(
              Expression::CreateAssignment(kInvalidExpressionId, right_id));
          summary.num_expressions_modified++;
        } else if (right_expr.IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(
              Expression::CreateAssignment(kInvalidExpressionId, left_id));
          summary.num_expressions_modified++;
        }
      } else if (expr.name() == "-") {
        if (left_expr.IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(Expression::CreateUnaryArithmetic("-", right_id));
          summary.num_expressions_modified++;
          continue;
        } else if (right_expr.IsCompileTimeConstantAndEqualTo(0)) {
          expr.Replace(
              Expression::CreateAssignment(kInvalidExpressionId, left_id));
          summary.num_expressions_modified++;
        }
      }
    }
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  summary.end();
  return summary;
}

inline OptimizationPassSummary ConstantFolding(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ConstantFolding";
  summary.start();
  ExpressionDependencies dep(*graph);
  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    auto& expr = graph->ExpressionForId(id);
    if (!expr.HasValidLhs()) continue;
    if (!dep.DataForExpressionId(expr.lhs_id()).IsSSA()) continue;
    if (expr.type() == ExpressionType::COMPILE_TIME_CONSTANT) continue;

    // Check if all args are compile time constants
    bool all_const = true;
    for (auto a : expr.arguments()) {
      if (graph->ExpressionForId(a).type() !=
          ExpressionType::COMPILE_TIME_CONSTANT) {
        all_const = false;
        break;
      }
    }
    if (!all_const) {
      continue;
    }

    // Evaluate + replace by compile time constant
    switch (expr.type()) {
      case ExpressionType::ASSIGNMENT: {
        auto& rhs_expr0 = graph->ExpressionForId(expr.arguments()[0]);
        expr.Replace(Expression::CreateCompileTimeConstant(rhs_expr0.value()));
      }
      case ExpressionType::UNARY_ARITHMETIC: {
        auto& rhs_expr0 = graph->ExpressionForId(expr.arguments()[0]);
        if (expr.name() == "+") {
          double value = +rhs_expr0.value();
          expr.Replace(Expression::CreateCompileTimeConstant(value));
          summary.num_expressions_modified++;
        } else if (expr.name() == "-") {
          double value = -rhs_expr0.value();
          expr.Replace(Expression::CreateCompileTimeConstant(value));
          summary.num_expressions_modified++;
        }
        break;
      }
      case ExpressionType::BINARY_ARITHMETIC: {
        auto& rhs_expr0 = graph->ExpressionForId(expr.arguments()[0]);
        auto& rhs_expr1 = graph->ExpressionForId(expr.arguments()[1]);
        if (expr.name() == "+") {
          double value = rhs_expr0.value() + rhs_expr1.value();
          expr.Replace(Expression::CreateCompileTimeConstant(value));
          summary.num_expressions_modified++;
        } else if (expr.name() == "-") {
          double value = rhs_expr0.value() - rhs_expr1.value();
          expr.Replace(Expression::CreateCompileTimeConstant(value));
          summary.num_expressions_modified++;
        } else if (expr.name() == "*") {
          double value = rhs_expr0.value() * rhs_expr1.value();
          expr.Replace(Expression::CreateCompileTimeConstant(value));
          summary.num_expressions_modified++;
        } else if (expr.name() == "/") {
          double value = rhs_expr0.value() / rhs_expr1.value();
          expr.Replace(Expression::CreateCompileTimeConstant(value));
          summary.num_expressions_modified++;
        }
      }
    }
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  summary.end();

  return summary;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_ELIMINATE_NOPS_H_
