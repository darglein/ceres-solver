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
// This file tests the Expression class. For each member function one test is
// included here.
//
#include "ceres/internal/autodiff.h"
//
#include "DisneyBRDF.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/codegen/internal/expression.h"
#include "common.h"
#include "compare_cost_functions.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

template <typename FunctorType, int kNumResiduals, int... Ns>
void test_functor() {
  FunctorType cost_function_generated;
  CostFunctionToFunctor<FunctorType> cost_functor;
  auto* cost_function_ad =
      new ceres::AutoDiffCostFunction<CostFunctionToFunctor<FunctorType>,
                                      kNumResiduals,
                                      Ns...>(&cost_functor);

  // Run N times with random values in the range [-1,1]
  for (int i = 0; i < 1; ++i) {
    ceres::internal::compare_cost_functions<kNumResiduals, Ns...>(
        &cost_function_generated, cost_function_ad, true);
  }
}

TEST(AutodiffCodeGen, InputOutputAssignment) {
  test_functor<test::DisneyBRDF, 3, 10>();
}

}  // namespace internal
}  // namespace ceres
