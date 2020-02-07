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
#ifndef CERES_PUBLIC_CODEGEN_OPTIMIZATION_PASS_H_
#define CERES_PUBLIC_CODEGEN_OPTIMIZATION_PASS_H_

#include <string>
#include <vector>

#include "ceres/codegen/internal/expression.h"
#include "ceres/codegen/internal/expression_graph.h"

namespace ceres {
namespace internal {

// Interface class for optimization passes.
class OptimizationPass {
 public:
  // Applies the optimization to the given graph (in-place). The returned value
  // is the cost change C(old) - C(new) achieved during the optimization. A cost
  // change of 0 means, that nothing was done at all. The optimizer usually
  // iterates until all optimization passes return 0.
  virtual double operator()(ExpressionGraph& graph) = 0;
};

// [OptimizationPass] NOP Cleanup
//
// Short Description:
//   Removes all expression with type==ExpressionType::NOP.
//
// Description:
//   Removing an expression not at the end requires reindexing of all later
//   expressions. Therefore, other optimization passes replace expression with
//   NOP instead of removing them. This optimization pass removes all NOPs back
//   to front.
//
// Example:
//   v_0 = 1;
//   // NOP
//   v_2 = 2;
//   v_3 = v_0 + v_2
// Transforms to:
//   v_0 = 1;
//   v_1 = 2;
//   v_2 = v_0 + v_1;
class NopCleanup : public OptimizationPass {
 public:
  virtual double operator()(ExpressionGraph& graph) override;
};

// [OptimizationPass] Dead Code Removal
//
// Short Description:
//   Removes unused expressions and branches by replacing them with NOPs.
//
// Description:
//  This module removes unused expressions back to front in a single pass.
//    for e in (graph.end...graph.begin)
//       if unused(e)
//          e = NOP
//  An expression is unused, if
//    - the left hand side variable is not referenced by later expressions
//    - and type!=OUTPUT_ASSIGNMENT
//  Unused branches are removed in a two-way procedure. First, empty else
//  blocks are removed. Second, if-blocks which are empty and
//  don't have an else-block removed.
//
//
// Example:
//   v_0 = 1;
//   v_1 = 3
//   v_2 = 3
//   v_3 = v_0 + v_1;
//   v_4 = v_1 + v_2
//   residuals[0] = v_3;
// Transforms to:
//   v_0 = 1;
//   v_1 = 3
//   // NOP
//   v_3 = v_0 + v_1;
//   // NOP
//   residuals[0] = v_3;
class DeadCodeRemoval : public OptimizationPass {
 public:
  virtual double operator()(ExpressionGraph& graph) override;

 private:
  bool unused(const ExpressionGraph& graph, ExpressionId id);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_CODE_GENERATOR_H_
