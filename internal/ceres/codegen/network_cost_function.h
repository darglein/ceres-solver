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

#include "ceres/codegen/codegen_cost_function.h"
#include "ceres/rotation.h"
namespace test {

constexpr int DenseLayerSize(int input, int output) {
  return output * (input + 1);
}

static constexpr int input_size = 784;
static constexpr int inner_size = 64;
static constexpr int output_size = 10;
static constexpr int inner_layers = 2;

static constexpr int nn_size =
    DenseLayerSize(input_size, inner_size) +
    (inner_layers * DenseLayerSize(inner_size, inner_size)) +
    DenseLayerSize(inner_size, output_size);

struct NeuralBACost
    : public ceres::CodegenCostFunction<2, 7, 3, 5 * DenseLayerSize(2, 2)> {
  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  const T* const network_params,
                  T* residuals) const {
    T ox = CERES_LOCAL_VARIABLE(T, observed_x);
    T oy = CERES_LOCAL_VARIABLE(T, observed_y);

    // camera[0,1,2] are the angle-axis rotation.
    T p[3];

    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    T x[2] = {xp, yp};
    T tmp[2];

    int current_params = 0;
    for (int i = 0; i < 5; ++i) {
      DenseLayer(network_params + current_params, x, tmp, 2, 2);
      //      SigmoidArray(tmp, x, 2);
      for (int j = 0; j < 2; ++j) {
        x[j] = tmp[j];
      }
      //      current_params += DenseLayerSize(2, 2);
    }

    //    x[0] = tmp[0];
    //    x[1] = tmp[1];

    // Compute final projected point position.
    const T& focal = camera[6];
    const T predicted_x = focal * x[0];
    const T predicted_y = focal * x[1];

    // The error is the difference between the predicted and observed
    // position.
    residuals[0] = predicted_x - ox;
    residuals[1] = predicted_y - oy;

    return true;
  }

  double observed_x;
  double observed_y;

  template <typename T>
  void DenseLayer(const T* params,
                  const T* input,
                  T* output,
                  int input_size,
                  int output_size) const {
    for (int i = 0; i < output_size; ++i) {
      T weighted_sum = T(0);
      for (int j = 0; j < input_size; ++j) {
        weighted_sum += params[i * (input_size + 1) + j] * input[j];
      }
      // add bias
      //      weighted_sum += params[i * (input_size + 1) + input_size];

      output[i] = weighted_sum;
      //      output[i] = input[i];
    }
  }

  template <typename T>
  void SigmoidArray(const T* input, T* output, int N) const {
    for (int i = 0; i < N; ++i) {
      output[i] = T(1.0) / (T(1.0) + exp(-input[i]));
    }
  }

  template <typename T>
  void SoftMaxArray(const T* input, T* output, int N) const {
    T exp_sum = T(0);
    for (int i = 0; i < N; ++i) {
      exp_sum += exp(input[i]);
    }

    for (int i = 0; i < N; ++i) {
      output[i] = exp(input[i]) / exp_sum;
    }
  }

#include "tests/neuralbacost.h"
};  // namespace test

}  // namespace test
