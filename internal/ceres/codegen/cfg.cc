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
//
// This file tests the ExpressionGraph class. This test depends on the
// correctness of Expression.
//
#include "ceres/codegen/internal/cfg.h"

#include <algorithm>

#include "glog/logging.h"
namespace ceres {
namespace internal {

CFG::CFG(const ExpressionGraph& graph) : nodes_(graph.Size()) {
  for (ExpressionId id = 0; id < graph.Size(); ++id) {
    auto& expr = graph.ExpressionForId(id);
    auto& node = nodes_[id];
    node.id = id;

    if (expr.type() == ExpressionType::IF) {
      // From an if we can either jump to the first expression inside the "true"
      // block or the first expression inside the "false" block. If there is no
      // "false" block the first expression after endif is used.
      node.outgoing_edges.push_back(id + 1);
      auto else_expr = graph.FindMatchingElse(id);
      if (else_expr == kInvalidExpressionId) {
        node.outgoing_edges.push_back(graph.FindMatchingEndif(id) + 1);
      } else {
        node.outgoing_edges.push_back(else_expr + 1);
      }
    } else if (expr.type() == ExpressionType::ELSE) {
      // Else is reached only from the "true" block. Therefore we have to jump
      // to the first expression after the matching endif.
      node.outgoing_edges.push_back(graph.FindMatchingEndif(id) + 1);
    } else if (expr.type() == ExpressionType::ENDIF) {
      // Endif is reached from the inside and just stepped over.
      node.outgoing_edges.push_back(id + 1);
    } else {
      // All other expression just go to the next.
      node.outgoing_edges.push_back(id + 1);
    }
  }

  // Insert incoming edges and mark jump to the end with kInvalidIndex.
  for (ExpressionId id = 0; id < graph.Size(); ++id) {
    auto& node = nodes_[id];
    for (auto& target : node.outgoing_edges) {
      CHECK(id != target) << "An expression cannot jump to itself.";
      if (target == Size()) {
        target = kInvalidExpressionId;
      } else {
        nodes_[target].incoming_edges.push_back(id);
      }
    }
  }

  // Sort incoming and outgoing edges by id
  for (ExpressionId id = 0; id < graph.Size(); ++id) {
    auto& node = nodes_[id];
    std::sort(node.incoming_edges.begin(), node.incoming_edges.end());
    std::sort(node.outgoing_edges.begin(), node.outgoing_edges.end());
  }
}

}  // namespace internal
}  // namespace ceres
