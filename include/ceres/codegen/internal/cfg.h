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
#ifndef CERES_PUBLIC_CODEGEN_CFG_H_
#define CERES_PUBLIC_CODEGEN_CFG_H_

#include <string>
#include <vector>

#include "ceres/codegen/internal/expression.h"
#include "ceres/codegen/internal/expression_graph.h"

namespace ceres {
namespace internal {

// The control flow graph (CFG) for an ExpressionGraph.
//
//
// https://en.wikipedia.org/wiki/Control-flow_graph

using BlockId = int;
static constexpr BlockId kEndNode = -1;
static constexpr BlockId kStartNode = 0;

class CFG {
 public:
  struct BasicBlock {
    BlockId id;
    ExpressionId start, end;
    std::vector<BlockId> outgoing_edges;
    std::vector<BlockId> incoming_edges;

    BasicBlock() = default;

    size_t Size() const { return end - start; }
  };

  // Creates the CFG for the given graph. If 'graph' is changed, this CFG is
  // invalid.
  CFG(const ExpressionGraph& graph);

  BlockId Idom(BlockId id) const { return doms_[id]; }

  // Returns true if a dominates b.
  // a is always executed before b.
  bool DominateExpression(ExpressionId a, ExpressionId b);

  bool DominateBlock(BlockId a, BlockId b);

  const BasicBlock& NodeForId(BlockId id) const { return blocks_[id]; }
  int Size() const { return blocks_.size(); }
  BlockId BlockIdForExpressionId(ExpressionId id) {
    return block_for_expression_[id];
  }

  void Print() const;

  std::vector<BlockId> PostOrder(BlockId start = 0);

  std::vector<ExpressionId> possibleValuesOfExpressionAtLocation(
      ExpressionGraph& graph, ExpressionId location, ExpressionId id);

 private:
  std::vector<BlockId> doms_;
  std::vector<BasicBlock> blocks_;
  std::vector<BlockId> block_for_expression_;

  void PostOrder(BlockId block,
                 std::vector<BlockId>* result,
                 std::vector<bool>* visisted);
};
}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_CODE_GENERATOR_H_
