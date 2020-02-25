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
#ifndef CERES_PUBLIC_CODEGEN_EXPRESSION_DEPENDENCIES_H_
#define CERES_PUBLIC_CODEGEN_EXPRESSION_DEPENDENCIES_H_

#include <iostream>
#include <vector>

#include "ceres/codegen/internal/expression.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "glog/logging.h"
namespace ceres {
namespace internal {

class ExpressionDependencies {
 public:
  struct Data {
    // All other expressions that write to the left hand side of this
    // expression.
    std::vector<ExpressionId> written_to;

    // All other expressions that use the left hand side of this expression as
    // an argument.
    std::vector<ExpressionId> used_by;

    // An expression is in SSA form if the left hand side is written to exactly
    // once.
    bool IsSSA() const { return written_to.size() == 1; }

    bool Unused() const { return used_by.empty(); }
  };
  ExpressionDependencies(const ExpressionGraph& graph);

  const Data& DataForExpressionId(ExpressionId id) {
    CHECK(id != kInvalidExpressionId) << id;
    CHECK(graph_->ExpressionForId(id).lhs_id() != kInvalidExpressionId)
        << graph_->ExpressionForId(id).lhs_id() << " " << id;
    CHECK(graph_->ExpressionForId(id).lhs_id() == id)
        << graph_->ExpressionForId(id).lhs_id() << " " << id;
    return data_[id];
  }

 private:
  std::vector<Data> data_;
  const ExpressionGraph* graph_;
};
}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_EXPRESSION_DEPENDENCIES_H_
