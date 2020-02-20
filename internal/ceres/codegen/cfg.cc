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
#include <iostream>

#include "glog/logging.h"
namespace ceres {
namespace internal {

CFG::CFG(const ExpressionGraph& graph)
    : block_for_expression_(graph.Size(), kInvalidExpressionId) {
  // Insert the start block
  BlockId currentBlockId = 0;
  blocks_.push_back({0, 0, 0});

  for (ExpressionId id = 0; id < graph.Size(); ++id) {
    auto& expr = graph.ExpressionForId(id);
    auto& node = blocks_[currentBlockId];
    bool start_new_block = false;

    block_for_expression_[id] = currentBlockId;
    node.end++;

    if (expr.type() == ExpressionType::IF) {
      // From an if we can either jump to the first expression inside the "true"
      // block or the first expression inside the "false" block. If there is no
      // "false" block the first expression after endif is used.
      start_new_block = true;
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
      start_new_block = true;
      node.outgoing_edges.push_back(graph.FindMatchingEndif(id) + 1);
    } else if (expr.type() == ExpressionType::ENDIF) {
      // Endif is reached from the inside and just stepped over.
      start_new_block = true;
      node.outgoing_edges.push_back(id + 1);
    } else {
      // All other expression just go to the next.
      node.outgoing_edges.push_back(id + 1);
    }

    if (start_new_block) {
      blocks_.push_back({++currentBlockId, id + 1, id + 1});
    }
  }

  // Remove empty blocks
  blocks_.erase(
      std::remove_if(blocks_.begin(),
                     blocks_.end(),
                     [](const BasicBlock& b) { return b.Size() == 0; }),
      blocks_.end());
  if (blocks_.empty()) return;

  for (auto& block : blocks_) {
    // Copy edges
    auto edges = block.outgoing_edges;

    // Convert from expression ids to block ids
    block.outgoing_edges.clear();

    for (auto e : edges) {
      if (e == graph.Size()) {
        block.outgoing_edges.push_back(kEndNode);
        continue;
      }
      auto target_block_id = BlockIdForExpressionId(e);
      if (target_block_id != block.id) {
        block.outgoing_edges.push_back(target_block_id);
        blocks_[target_block_id].incoming_edges.push_back(block.id);
      }
    }
  }

  for (auto& block : blocks_) {
    std::sort(block.outgoing_edges.begin(), block.outgoing_edges.end());
    std::sort(block.incoming_edges.begin(), block.incoming_edges.end());
  }

  //  ===== Build dominator tree ====

  auto intersect = [this](BlockId b1, BlockId b2) {
    auto finger1 = b1;
    auto finger2 = b2;
    while (finger1 != finger2) {
      while (finger1 < finger2) {
        finger1 = doms_[finger1];
      }
      while (finger2 < finger1) {
        finger2 = doms_[finger2];
      }
    }
    return finger1;
  };

  // 1. Initialize
  const int start_node = 0;
  doms_.resize(blocks_.size(), -1);

  // 2. compute reverse post order of the CFG without the start node
  std::vector<BlockId> post_order_to_node = PostOrder(start_node);
  std::vector<BlockId> node_to_post_order(post_order_to_node.size());
  for (int i = 0; i < post_order_to_node.size(); ++i) {
    node_to_post_order[post_order_to_node[i]] = i;
  }

  doms_[node_to_post_order[start_node]] = node_to_post_order[start_node];

  for (auto r : post_order_to_node) {
    //    std::cout << r << std::endl;
  }

  bool changed = true;
  while (changed) {
    changed = false;
    // Iterate over blocks in reverse post order. Skip start node.
    for (BlockId b = post_order_to_node.size() - 2; b >= 0; --b) {
      //      std::cout << "post order " << b << std::endl;
      // auto b = reverse_post_order[post_order_id];
      auto node = post_order_to_node[b];
      auto new_idom = kEndNode;
      for (auto pred_node : blocks_[node].incoming_edges) {
        auto pred = node_to_post_order[pred_node];
        //        std::cout << "pred " << pred << std::endl;
        if (pred > b) {
          new_idom = pred;
          break;
        }
      }
      CHECK(new_idom != kEndNode);

      for (auto pred_node : blocks_[node].incoming_edges) {
        auto p = node_to_post_order[pred_node];
        if (p == new_idom) continue;
        if (doms_[p] != -1) {
          new_idom = intersect(p, new_idom);
        }
      }
      if (doms_[b] != new_idom) {
        //        std::cout << "set " << b << " = " << new_idom << std::endl;
        doms_[b] = new_idom;
        changed = true;
      }
    }
  }

  // invert doms to match actual block ids
  auto cpy = doms_;
  for (int i = 0; i < cpy.size(); ++i) {
    doms_[i] = post_order_to_node[cpy[node_to_post_order[i]]];
    //    doms_[node_to_post_order[i]] = cpy[i];
  }

  for (auto d : doms_) {
    //    std::cout << "dom " << d << std::endl;
  }
}

bool CFG::DominateExpression(ExpressionId a, ExpressionId b) {
  auto block_a = BlockIdForExpressionId(a);
  auto block_b = BlockIdForExpressionId(b);

  if (block_a == block_b) {
    return a <= b;
  }

  return DominateBlock(block_a, block_b);
}

bool CFG::DominateBlock(BlockId a, BlockId b) {
  // go upwards from b until we find either a or the start.
  while (true) {
    if (a == b) {
      return true;
    }

    if (b == kStartNode) {
      return false;
    }

    b = doms_[b];
  }
}

void CFG::Print() const {
  for (auto block : blocks_) {
    std::cout << "[block] " << block.id << " Expr(" << block.start << " "
              << block.end << ") Outgoing {";
    for (auto e : block.outgoing_edges) {
      std::cout << e << ", ";
    }
    std::cout << " }" << std::endl;
  }
}

std::vector<BlockId> CFG::PostOrder(BlockId start) {
  std::vector<BlockId> result;
  std::vector<bool> visisted(Size(), false);
  PostOrder(start, &result, &visisted);
  return result;
}

void CFG::PostOrder(BlockId block,
                    std::vector<BlockId>* result,
                    std::vector<bool>* visisted) {
  if (block == kEndNode) return;
  if ((*visisted)[block]) return;
  (*visisted)[block] = true;
  for (auto edge : blocks_[block].outgoing_edges) {
    PostOrder(edge, result, visisted);
  }
  result->push_back(block);
}

}  // namespace internal
}  // namespace ceres
