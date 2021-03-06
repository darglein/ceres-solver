// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
// A test comparing the result of AutoDiffCodeGen to AutoDiff.

#include <memory>

#include "ceres/array_utils.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "compare_cost_functions.h"
#include "gtest/gtest.h"
#include "snavely_reprojection_error.h"
#include "test_utils.h"
namespace ceres {

namespace examples {

TEST(AutodiffCodeGen, SnavelyReprojectionError) {
  CostFunction* cost_function = new test::SnavelyReprojectionErrorGen();
  using CostFunctorType =
      internal::CostFunctionToFunctor<test::SnavelyReprojectionErrorGen>;
  CostFunction* cost_function_ad =
      new ceres::AutoDiffCostFunction<CostFunctorType, 2, 9, 3>(
          new CostFunctorType());
  ceres::internal::CompareCostFunctions(cost_function, cost_function_ad, 1, 0);
}

}  // namespace examples
}  // namespace ceres
