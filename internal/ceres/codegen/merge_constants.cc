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
#include "ceres/codegen/internal/merge_constants.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "ceres/codegen/internal/expression_dependencies.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

static OptimizationPassSummary ConstantsToSSA(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ConstantsToSSA";
  summary.start();
  ExpressionDependencies dep(*graph);

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    auto& expr = graph->ExpressionForId(id);
    if (expr.type() != ExpressionType::COMPILE_TIME_CONSTANT) {
      continue;
    }

    if (dep.DataForExpressionId(expr.lhs_id()).IsSSA()) {
      // already SSA -> do nothing
      continue;
    }

    // Insert compile time constant
    graph->Insert(id, Expression::CreateCompileTimeConstant(expr.value()));

    // Add assignemnt from the constant to the current value
    graph->ExpressionForId(id + 1).Replace(
        Expression::CreateAssignment(kInvalidExpressionId, id));

    // After insertion rebuild is mandatory!
    dep.Rebuild();

    summary.num_expressions_inserted++;
    summary.num_expressions_modified++;
    summary.expression_graph_changed = true;
  }

  summary.end();
  return summary;
}  // namespace internal

// The "Constant Section" of an ExpressionGraph is the index range at the
// beginning of the program which contains only COMPILE_TIME_CONSTANT
// expressions. The end of this section is therefore the first Expression from
// the beginning that is not COMPILE_TIME_CONSTANT.
inline ExpressionId EndOfConstantSection(const ExpressionGraph& graph) {
  for (ExpressionId id = 0; id < graph.Size(); ++id) {
    if (graph.ExpressionForId(id).type() !=
        ExpressionType::COMPILE_TIME_CONSTANT)
      return id;
  }
  return graph.Size();
}

// [OptimizationPass] Move Compile Time Constants To The Beginning
//
// Short Description:
//   Moves compile time constants to the start of the program.
//
// Description:
//   Having all compile time constants at the beginning of a program is
//   beneficial, because other optimizations passes don't have to check scoping,
//   dominance, and definition order.
//
// Example:
//   v_0 = 1;
//   v_1 = v_0 + v_0;
//   v_2 = 42;
//   if ( v_0 < v_2 ) {
//      v_4 = 6;
//      v_3 = v_4 + v_2;
//   }
//   v_6 = v_0 + v_2
//
// Transforms to:
//   v_0 = 1;
//   v_1 = 42;
//   v_2 = 6;
//   v_3 = v_0 + v_0;
//   // NOP
//   if ( v_0 < v_3 ) {
//      // NOP
//      v_4 = v_2 + v_3;
//   }
//   v_6 = v_0 + v_3
//
static OptimizationPassSummary MoveConstantsToBeginning(
    ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ReorderCompileTimeConstants";
  summary.start();
  ExpressionId end_of_constant_section = EndOfConstantSection(*graph);

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
  summary.end();
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
static OptimizationPassSummary MergeCompileTimeConstants(
    ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "MergeCompileTimeConstants";

  summary.start();
  ExpressionId end_of_constant_section = EndOfConstantSection(*graph);

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
  summary.end();
  return summary;
}

// Merge compile time constants of the same value.
// Only merges values from the constant section
static OptimizationPassSummary SortCompileTimeConstants(
    ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "SortCompileTimeConstants";

  summary.start();
  ExpressionId end_of_constant_section = EndOfConstantSection(*graph);

  std::vector<std::pair<double, int>> exprs;
  std::vector<ExpressionId> source_id;

  for (ExpressionId id = 0; id < end_of_constant_section; ++id) {
    auto& expr = graph->ExpressionForId(id);
    exprs.emplace_back(expr.value(), id);
    source_id.push_back(id);
  }

  //  auto new_exprs = exprs;
  std::sort(exprs.begin(), exprs.end());

  std::vector<ExpressionId> target_id(end_of_constant_section);

  for (ExpressionId id = 0; id < end_of_constant_section; ++id) {
    auto& expr = graph->ExpressionForId(id);
    expr.Replace(Expression::CreateCompileTimeConstant(exprs[id].first));

    target_id[exprs[id].second] = id;
  }

  //  for (ExpressionId id = 0; id < end_of_constant_section; ++id) {
  //    std::cout << source_id[id] << " " << target_id[id] << std::endl;
  //  }

  for (ExpressionId i = end_of_constant_section; i < graph->Size(); ++i) {
    auto& expr = graph->ExpressionForId(i);

    expr.UpdateId(source_id, target_id);
  }

  summary.end();
  return summary;
}  // namespace internal

OptimizationPassSummary MergeConstants(ExpressionGraph* graph) {
  ConstantsToSSA(graph);

  OptimizationPassSummary summary1 = MoveConstantsToBeginning(graph);
  OptimizationPassSummary summary2 = MergeCompileTimeConstants(graph);
  SortCompileTimeConstants(graph);
  //  OptimizationPassSummary combined;
  //  combined.expression_graph_changed =
  //      summary1.expression_graph_changed | summary2.expression_graph_changed;
  //  combined.num_expressions_modified =
  //      summary1.num_expressions_modified + summary2.num_expressions_modified;
  return OptimizationPassSummary();
}

}  // namespace internal
}  // namespace ceres
