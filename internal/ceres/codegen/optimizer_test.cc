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
#define CERES_CODEGEN

#include "ceres/codegen/internal/optimizer.h"

#include "ceres/codegen/internal/code_generator.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/expression_ref.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

static void GenerateAndCheck(const ExpressionGraph& graph,
                             const std::vector<std::string>& reference) {
  Optimizer::Options optimizer_options;
  Optimizer optimizer(optimizer_options);
  auto opt = optimizer.run(graph);

  CodeGenerator::Options generator_options;
  CodeGenerator gen(opt, generator_options);
  auto code = gen.Generate();
  //  EXPECT_EQ(code.size(), reference.size());

  for (int i = 0; i < code.size(); ++i) {
    std::cout << code[i] << std::endl;
    //    EXPECT_EQ(code[i], reference[i]) << "Invalid Line: " << (i + 1);
  }
}

using T = ExpressionRef;

// Now we add one TEST for each ExpressionType.
TEST(CodeGenerator, COMPILE_TIME_CONSTANT) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(123.5);
  MakeOutput(b, "residual[0]");

  auto graph = StopRecordingExpressions();
  GenerateAndCheck(graph, {});
}
}  // namespace internal
}  // namespace ceres
