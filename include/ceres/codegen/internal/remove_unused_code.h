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

#include <algorithm>
#include <iostream>
#include <set>

#include "ceres/codegen/internal/cfg.h"
#include "ceres/codegen/internal/expression_dependencies.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/optimization_pass_summary.h"
#include "glog/logging.h"
namespace ceres {
namespace internal {

inline OptimizationPassSummary RemoveUnusedCode(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "RemoveUnusedCode";
  summary.start();
  ExpressionDependencies dep(*graph);

  for (ExpressionId id = graph->Size() - 1; id >= 0; --id) {
    Expression& expr = graph->ExpressionForId(id);
    if (expr.type() == ExpressionType::OUTPUT_ASSIGNMENT ||
        expr.type() == ExpressionType::RETURN) {
      continue;
    }

    if (expr.HasValidLhs() && dep.DataForExpressionId(expr.lhs_id()).Unused()) {
      for (auto e : dep.DataForExpressionId(expr.lhs_id()).written_to) {
        graph->ExpressionForId(e).MakeNop();
        summary.num_expressions_replaced_by_nop++;
      }
      dep.Rebuild();
    }
  }

  // Remove non ssa expression if there is no use after this assignemnt.
  //
  // v_0 = 1;
  // v_1 = v_0;
  // v_0 = 2;
  //
  // Here the third expression can be removed even tho v_0 is not completely
  // unused.
  for (ExpressionId id = graph->Size() - 1; id >= 0; --id) {
    Expression& expr = graph->ExpressionForId(id);
    if (expr.type() == ExpressionType::OUTPUT_ASSIGNMENT ||
        expr.type() == ExpressionType::RETURN || !expr.HasValidLhs()) {
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
      dep.Rebuild();
    }
  }

  summary.expression_graph_changed =
      summary.num_expressions_replaced_by_nop > 0;
  summary.end();
  return summary;
}
inline OptimizationPassSummary ForwardFlow(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ForwardFlow";
  summary.start();

  ExpressionDependencies expr_dep(*graph);
  CFG cfg(*graph);

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    Expression& expr = graph->ExpressionForId(id);
    if (!expr.HasValidLhs()) {
      continue;
    }
    auto& dep = expr_dep.DataForExpressionId(expr.lhs_id());
    if (dep.IsSSA()) {
      continue;
    }
    if (expr.lhs_id() != id) {
      continue;
    }
    //    std::cout << "forward flow " << id << std::endl;

    std::vector<std::vector<ExpressionId>> reachable_writes;
    for (auto w : dep.used_by) {
      auto res = cfg.possibleValuesOfExpressionAtLocation(*graph, w, id);
      reachable_writes.push_back(res);
      // if there is only one possible result just use it.

      //      std::cout << "res " << w << ": ";
      //      for (auto r : res) {
      //        std::cout << r << ",";
      //      }
      //      std::cout << std::endl;
    }

    // let's combine them and look if some writes are unreachable.
    std::set<ExpressionId> all_reachable;
    for (auto v : reachable_writes) {
      all_reachable.insert(v.begin(), v.end());
    }

    //    if (all_reachable.empty()) {
    //      std::cout << "empty" << std::endl;
    //      continue;
    //    }

    CHECK(!all_reachable.empty());

    bool deleted_something = false;
    for (auto v : dep.written_to) {
      if (all_reachable.find(v) == all_reachable.end()) {
        //        std::cout << "found unused assignment " << v << std::endl;
        graph->ExpressionForId(v).MakeNop();
        deleted_something = true;
      }
    }

    if (deleted_something) {
      // find a new id for everything
      auto new_id = *all_reachable.begin();

      for (auto v : dep.written_to) {
        graph->ExpressionForId(v).UpdateId(id, new_id);
      }
      for (auto v : dep.used_by) {
        graph->ExpressionForId(v).UpdateId(id, new_id);
      }
      summary.num_expressions_modified++;
      expr_dep.Rebuild();
      continue;
    }

    // didn't delete anything here.
    for (int i = 0; i < reachable_writes.size(); ++i) {
      auto used_id = dep.used_by[i];
      auto writes = reachable_writes[i];

      //      if(writes.size() == 1)
      //      {
      //        auto ex = writes[]
      //      }

      if (writes.size() > 1) {
        auto first_ex = writes[0];
        auto first_expr = graph->ExpressionForId(first_ex);

        CHECK(first_expr.type() == ExpressionType::ASSIGNMENT);

        auto acutal_id = first_expr.arguments()[0];
        if (!expr_dep.DataForExpressionId(acutal_id).IsSSA()) {
          continue;
        }
        //        std::cout << "used " << used_id << " " << first_ex <<
        //        std::endl;

        bool found = true;
        for (auto a : writes) {
          auto sadf = graph->ExpressionForId(a);
          CHECK(sadf.type() == ExpressionType::ASSIGNMENT);
          if (sadf.arguments()[0] != acutal_id) {
            found = false;
            break;
          }
        }

        if (found) {
          //          std::cout << "update " << id << first_ex << std::endl;
          graph->ExpressionForId(used_id).UpdateId(id, acutal_id);
          summary.num_expressions_modified++;
        }
      }
    }
  }

  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  summary.end();
  return summary;
}

inline OptimizationPassSummary ToPartialSSA(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "ToPartialSSA";
  summary.start();

  ExpressionDependencies expr_dep(*graph);
  CFG cfg(*graph);

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    Expression& expr = graph->ExpressionForId(id);
    if (!expr.HasValidLhs()) {
      continue;
    }

    auto& dep = expr_dep.DataForExpressionId(expr.lhs_id());
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
        expr_dep.Rebuild();
      }
      continue;
    }

    //    std::cout << "convert to ssa " << id << std::endl;
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  summary.end();
  return summary;
}

inline OptimizationPassSummary TrivialAssignmentElimination(
    ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "TrivialAssignmentElimination";

  summary.start();
  ExpressionDependencies deps(*graph);

  for (ExpressionId id = graph->Size() - 1; id >= 0; --id) {
    Expression& expr = graph->ExpressionForId(id);
    if (!expr.HasValidLhs()) {
      continue;
    }
    auto dep = deps.DataForExpressionId(expr.lhs_id());

    //    if (expr.type() == ExpressionType::ASSIGNMENT) {
    //      auto& target_expr = graph->ExpressionForId(expr.arguments()[0]);
    //      if (dep.IsSSA() &&
    //          target_expr.type() == ExpressionType::COMPILE_TIME_CONSTANT) {
    //        CHECK(deps.DataForExpressionId(expr.arguments()[0]).IsSSA());
    //        expr.Replace(
    //            Expression::CreateCompileTimeConstant(target_expr.value()));
    //        summary.num_expressions_modified++;
    //        deps.Rebuild();
    //        continue;
    //      }
    //    }

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

      if (other_expr.arguments().empty()) continue;
      if (!deps.DataForExpressionId(p).IsSSA()) continue;

      //      CHECK(!other_expr.arguments().empty());
      auto new_target = other_expr.arguments()[0];
      if (!deps.DataForExpressionId(new_target).IsSSA()) continue;

      expr.UpdateId(p, new_target);
      summary.num_expressions_modified++;
      deps.Rebuild();
    }
  }
  summary.expression_graph_changed = summary.num_expressions_modified > 0;
  summary.end();
  return summary;
}

struct OrderingResult {
  bool in_order;
  bool leaf;
  std::vector<ExpressionId> leaf_nodes;
  std::vector<ExpressionId> internal_nodes;

  ExpressionId largestLeaf() {
    if (leaf_nodes.empty()) return kInvalidExpressionId;
    return *std::max_element(leaf_nodes.begin(), leaf_nodes.end());
  }
};

inline std::ostream& operator<<(std::ostream& strm, const OrderingResult& res) {
  strm << "Ordering (" << res.in_order << "," << res.leaf << ", "
       << res.leaf_nodes.size() << "," << res.internal_nodes.size() << ")";
  return strm;
}

inline OrderingResult MakeLinearOrder(ExpressionGraph& graph,
                                      ExpressionDependencies& deps,
                                      ExpressionId id,
                                      bool forward,
                                      const std::string& v) {
  auto& expr = graph.ExpressionForId(id);

  if (!expr.IsMultiplication(v)) {
    // Non- multiplications are always inputs (leafs)
    return {true, true, {id}, {}};
  }

  if (!deps.DataForExpressionId(id).IsSSA() ||
      deps.DataForExpressionId(id).used_by.size() > 1) {
    // Use non-SSA and multi-used expressions as leafs
    //    std::cout << "non ssa " << id << std::endl;
    return {true, true, {id}, {}};
  }

  // From here on we are not leaf anymore

  // order children
  auto res_left = MakeLinearOrder(graph, deps, expr.arguments()[0], forward, v);
  auto res_right =
      MakeLinearOrder(graph, deps, expr.arguments()[1], forward, v);

  //  std::cout << id << " left: " << res_left << std::endl;
  //  std::cout << id << " right: " << res_right << std::endl;

  CHECK(res_left.in_order);
  CHECK(res_right.in_order);

  std::vector<ExpressionId> all_leafs = res_left.leaf_nodes;
  all_leafs.insert(all_leafs.end(),
                   res_right.leaf_nodes.begin(),
                   res_right.leaf_nodes.end());

  std::vector<ExpressionId> all_inner = res_left.internal_nodes;
  all_inner.insert(all_inner.end(),
                   res_right.internal_nodes.begin(),
                   res_right.internal_nodes.end());

  if (res_left.leaf && res_right.leaf) {
    // just swap in place if needed
    if (res_left.leaf_nodes[0] > res_right.leaf_nodes[0] == forward) {
      std::swap((*expr.mutable_arguments())[0], (*expr.mutable_arguments())[1]);
      //      std::cout << "swap " << id << std::endl;
      return {true, false, all_leafs, all_inner};
    }
    return {true, false, all_leafs, all_inner};
  }

  if (forward) {
    if (std::is_sorted(
            all_leafs.begin(), all_leafs.end(), std::less<ExpressionId>())) {
      return {true, false, all_leafs, all_inner};
    }
    std::sort(all_leafs.begin(), all_leafs.end(), std::less<ExpressionId>());
  } else {
    if (std::is_sorted(
            all_leafs.begin(), all_leafs.end(), std::greater<ExpressionId>())) {
      return {true, false, all_leafs, all_inner};
    }
    std::sort(all_leafs.begin(), all_leafs.end(), std::greater<ExpressionId>());
  }

  auto current_id = id;

  //  graph.Insert(current_id, Expression::CreateComment("start"));
  //  current_id++;

  graph.Insert(
      current_id,
      Expression::CreateBinaryArithmetic(v, all_leafs[0], all_leafs[1]));
  current_id++;

  for (int i = 2; i < all_leafs.size(); ++i) {
    graph.Insert(
        current_id,
        Expression::CreateBinaryArithmetic(v, current_id - 1, all_leafs[i]));
    current_id++;
  }

  // replace the actual exprsesion with an assignemnt
  graph.ExpressionForId(current_id)
      .Replace(
          Expression::CreateAssignment(kInvalidExpressionId, current_id - 1));

  //  graph.Insert(current_id, Expression::CreateComment("end"));
  //  current_id++;
  //  CHECK(false);

  deps.Rebuild();
  return {false, false, all_leafs, all_inner};

#if 0
  // this is an valid inner id
  all_inner.insert(id);

  std::cout << id << " left: " << res_left << std::endl;
  std::cout << id << " right: " << res_right << std::endl;
  CHECK_EQ(all_leafs.size(), all_inner.size() + 1);

  // Merge ordering

  // Both children are leafs -> just order them
  if (res_left.leaf && res_right.leaf) {
    if (expr.arguments()[0] > expr.arguments()[1]) {
      std::swap((*expr.mutable_arguments())[0], (*expr.mutable_arguments())[1]);
    }
    return {true, false, all_leafs, all_inner};
  }

  // right is leaf and right has higher id then left
  // -> this is perfectly fine
  //  if (res_right.leaf &&
  //      res_left.largestLeaf() < res_right.largestLeaf()) {
  //    return {true, false, all_leafs, all_inner};
  //  }

  //  std::cout << "invalid order detected." << std::endl;
  //  std::cout << "leafs" << std::endl;
  //  for (auto l : all_leafs) {
  //    std::cout << l << std::endl;
  //  }
  //  std::cout << "inners" << std::endl;
  //  for (auto l : all_inner) {
  //    std::cout << l << std::endl;
  //  }

  std::vector<ExpressionId> leafs2(all_leafs.begin(), all_leafs.end());

  if (!forward) {
    std::reverse(leafs2.begin(), leafs2.end());
  }

#if 1
  // insert into existing expressions
  auto inner_it = all_inner.begin();
  auto leaf_it = leafs2.begin();

  // insert double leaf expr into first inner
  auto& inner_expr = graph.ExpressionForId(*inner_it);
  CHECK(inner_expr.IsMultiplication(v));
  (*inner_expr.mutable_arguments())[0] = *leaf_it++;
  (*inner_expr.mutable_arguments())[1] = *leaf_it++;

  // linearize the rest
  while (true) {
    auto new_inner_it = inner_it;
    ++new_inner_it;
    if (new_inner_it == all_inner.end()) {
      break;
    }
    auto& inner_expr = graph.ExpressionForId(*new_inner_it);
    CHECK(inner_expr.IsMultiplication(v));
    (*inner_expr.mutable_arguments())[0] = *inner_it;
    (*inner_expr.mutable_arguments())[1] = *leaf_it++;
    inner_it = new_inner_it;
  }

  CHECK(leaf_it == leafs2.end());
#else
  graph
      .Insert()

#endif
  deps.Rebuild();

  return {true, false, all_leafs, all_inner};
#endif
}  // namespace internal

inline OptimizationPassSummary SortArguments(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "SortArguments";
  summary.start();

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    Expression& expr = graph->ExpressionForId(id);
    if (expr.type() == ExpressionType::BINARY_ARITHMETIC) {
      if (expr.name() == "+" || expr.name() == "*") {
        if (expr.arguments()[0] > expr.arguments()[1]) {
          std::swap((*expr.mutable_arguments())[0],
                    (*expr.mutable_arguments())[1]);
          summary.num_expressions_modified++;
          summary.expression_graph_changed = true;
        }
      }
    }
  }

  summary.end();
  return summary;
}

inline bool CheckForwardArguments(ExpressionGraph* graph) {
  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    Expression& expr = graph->ExpressionForId(id);
    for (auto a : expr.arguments()) {
      if (a >= id) {
        std::cout << "invalid forward arg " << id << " " << (int)expr.type()
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

inline OptimizationPassSummary Reorder(ExpressionGraph* graph,
                                       bool forward,
                                       const std::string& v) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "Reorder";
  summary.start();

  //  SortArguments(graph);

  ExpressionDependencies deps(*graph);
  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    //    std::cout << "linearize " << id << std::endl;
    auto res = MakeLinearOrder(*graph, deps, id, forward, v);
    //    CHECK(res.in_order);
    if (!res.in_order) {
      //      break;
    }
  }

  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    auto res = MakeLinearOrder(*graph, deps, id, forward, v);
    CHECK(res.in_order);
  }

  CHECK(CheckForwardArguments(graph));
  summary.end();
  return summary;
}

inline OptimizationPassSummary MoveToUsage(ExpressionGraph* graph) {
  OptimizationPassSummary summary;
  summary.optimization_pass_name = "MoveToUsage";
  summary.start();

  ExpressionDependencies deps(*graph);
  for (ExpressionId id = 0; id < graph->Size(); ++id) {
    Expression& expr = graph->ExpressionForId(id);

    if (expr.type() == ExpressionType::COMPILE_TIME_CONSTANT) {
      //      continue;
    }

    //    std::cout << "move to " << expr.HasValidLhs() << std::endl;

    if (!expr.HasValidLhs()) {
      continue;
    }

    // only move single use ssa expressions
    auto& dep = deps.DataForExpressionId(expr.lhs_id());

    //    std::cout << "move to " << id << " " << dep.IsSSA() << " "
    //              << dep.used_by.size() << std::endl;

    if (!dep.IsSSA() || dep.used_by.size() != 1) {
      continue;
    }

    auto used_id = dep.used_by.front();

#if 1
    //    std::cout << "move to " << id << " " << used_id << std::endl;
    if (id < used_id) {
      continue;
      // check if all expressions in between are args from used
      // that would be fine too
      auto& used_expr = graph->ExpressionForId(used_id);
      auto args = used_expr.arguments();
      bool found = false;
      ExpressionId i = id;
      for (; i < used_id; ++i) {
        if (std::find(args.begin(), args.end(), i) == args.end()) {
          found = true;
          break;
        }
      }
      if (!found) {
        continue;
      }
    } else {
      //      std::cout << "bla" << std::endl;
    }
#else

    if (id == used_id + 1) {
      continue;
    }
#endif

    auto expr_cpy = expr;
    expr.MakeNop();

    expr_cpy.set_lhs_id(kInvalidExpressionId);
    graph->Insert(used_id, expr_cpy);

    auto& used_expr = graph->ExpressionForId(used_id + 1);
    used_expr.UpdateId(id, used_id);
    summary.num_expressions_replaced_by_nop++;
    summary.num_expressions_inserted++;
    summary.expression_graph_changed = true;

    deps.Rebuild();

    std::cout << "move " << id << " -> " << used_id << std::endl;
  }

  summary.end();
  return summary;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_ELIMINATE_NOPS_H_
