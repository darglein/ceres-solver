// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
// http://ceres-solver.org/
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
// Authors: sameeragarwal@google.com (Sameer Agarwal)

#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/ceres.h"
#include "ceres/jet.h"
#include "snavely_reprojection_error.h"

namespace ceres {

static void BM_BAAutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 6];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {jacobian1, jacobian2};

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<test::SnavelyReprojectionAD, 2, 6, 3>(
          new test::SnavelyReprojectionAD(x, y)));

  while (state.KeepRunning()) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

static void BM_BACodeGen(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 6];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {jacobian1, jacobian2};

  const double x = 0.2;
  const double y = 0.3;

  std::unique_ptr<ceres::CostFunction> cost_function(
      new test::SnavelyReprojectionErrorGen(x, y));

  while (state.KeepRunning()) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_BAAutoDiff)->ArgName("R")->Arg(0);
BENCHMARK(BM_BAAutoDiff)->ArgName("R+J")->Arg(1);
BENCHMARK(BM_BACodeGen)->ArgName("R")->Arg(0);
BENCHMARK(BM_BACodeGen)->ArgName("R+J")->Arg(1);

}  // namespace ceres

BENCHMARK_MAIN();
