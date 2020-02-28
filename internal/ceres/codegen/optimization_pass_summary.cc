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

#include "ceres/codegen/internal/optimization_pass_summary.h"

#include <iostream>

#include "ceres/wall_time.h"
#include "glog/logging.h"
namespace ceres {
namespace internal {

std::ostream& operator<<(std::ostream& strm,
                         const OptimizationPassSummary& summary) {
  strm << "[" << summary.optimization_pass_name << "]" << std::endl;
  strm << "   Changed         : " << summary.expression_graph_changed
       << std::endl;
  strm << "   Replaced by NOP : " << summary.num_expressions_replaced_by_nop
       << std::endl;
  strm << "   Removed         : " << summary.num_expressions_removed
       << std::endl;
  strm << "   Inserted        : " << summary.num_expressions_inserted
       << std::endl;
  strm << "   Modified        : " << summary.num_expressions_modified
       << std::endl;
  ;
  strm << "   Time            : " << summary.time;
  return strm;
}

void OptimizationPassSummary::start() {
  //  std::cout << "start " << optimization_pass_name << std::endl;
  time = WallTimeInSeconds();
}

void OptimizationPassSummary::end() {
  //  std::cout << "end" << std::endl;
  time = WallTimeInSeconds() - time;
}

OptimizationPassSummary operator+(const OptimizationPassSummary& a,
                                  const OptimizationPassSummary& b) {
  OptimizationPassSummary result = a;
  result.optimization_pass_name = "Sum";
  result.expression_graph_changed |= b.expression_graph_changed;
  result.num_expressions_replaced_by_nop += b.expression_graph_changed;
  result.num_expressions_removed += b.expression_graph_changed;
  result.num_expressions_inserted += b.expression_graph_changed;
  result.num_expressions_modified += b.expression_graph_changed;
  result.time += b.time;
  return result;
}

}  // namespace internal
}  // namespace ceres
