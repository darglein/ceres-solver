// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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
#ifndef CERES_INTERNAL_CERES_CODEGEN_COMPARE_H_
#define CERES_INTERNAL_CERES_CODEGEN_COMPARE_H_

#include "ceres/codegen/codegen_cost_function.h"
#include "ceres/random.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

inline void double_compare(double v1, double v2) {
  if (std::isnan(v2)) return;

  // If both are inf or nan it's fine too!
  if (std::isinf(v1) && std::isinf(v2)) {
    return;
  }

  if (std::isnan(v1) && std::isnan(v2)) {
    return;
  }

  EXPECT_NEAR(v1, v2, 1e-40);
}

template <int kNumResiduals, int... Ns>
void compare_cost_functions(CostFunction* f1,
                            CostFunction* f2,
                            bool random_values = true,
                            double value = 0) {
  using Params = StaticParameterDims<Ns...>;

  std::array<double, Params::kNumParameters> params_array;
  std::array<double*, Params::kNumParameters> params;

  for (auto& p : params_array) {
    if (random_values) {
      p = ceres::RandDouble() * 2.0 - 1.0;
    } else {
      p = value;
    }
  }
  for (int i = 0, k = 0; i < Params::kNumParameterBlocks;
       k += Params::GetDim(i), ++i) {
    params[i] = &params_array[k];
  }

  std::array<double, kNumResiduals> residuals_0, residuals_1;
  std::fill(residuals_0.begin(), residuals_0.end(), 0.0);
  std::fill(residuals_1.begin(), residuals_1.end(), 0.0);
  std::array<double, kNumResiduals * Params::kNumParameters> jacobians_array_0,
      jacobians_array_1;
  std::fill(jacobians_array_0.begin(), jacobians_array_0.end(), 0.0);
  std::fill(jacobians_array_1.begin(), jacobians_array_1.end(), 0.0);

  std::array<double*, Params::kNumParameterBlocks> jacobians_0, jacobians_1;

  for (int i = 0, k = 0; i < Params::kNumParameterBlocks;
       k += Params::GetDim(i), ++i) {
    jacobians_0[i] = &jacobians_array_0[k * kNumResiduals];
    jacobians_1[i] = &jacobians_array_1[k * kNumResiduals];
  }

  f1->Evaluate(params.data(), residuals_0.data(), jacobians_0.data());
  f2->Evaluate(params.data(), residuals_1.data(), jacobians_1.data());

  //  std::cout << kNumResiduals << std::endl;
  for (int i = 0; i < kNumResiduals; ++i) {
    double_compare(residuals_0[i], residuals_1[i]);
    //    std::cout << i << ": " << residuals_0[i] << " " << residuals_1[i]
    //              << std::endl;
  }
  //  std::cout << kNumResiduals * Params::kNumParameters << std::endl;
  for (int i = 0; i < kNumResiduals * Params::kNumParameters; ++i) {
    double_compare(jacobians_array_0[i], jacobians_array_1[i]);
    //    std::cout << i << ": " << jacobians_array_0[i] << " "
    //              << jacobians_array_1[i] << std::endl;
  }
}

}  // namespace internal
}  // namespace ceres
#endif  // CERES_INTERNAL_CERES_CODEGEN_COMMON_H_
