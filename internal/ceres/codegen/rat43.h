// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Templated struct implementing the camera model and residual
// computation for bundle adjustment used by Noah Snavely's Bundler
// SfM system. This is also the camera model/residual for the bundle
// adjustment problems in the BAL dataset. It is templated so that we
// can use Ceres's automatic differentiation to compute analytic
// jacobians.
//
// For details see: http://phototour.cs.washington.edu/bundler/
// and http://grail.cs.washington.edu/projects/bal/

#ifndef CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR3_H_
#define CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR3_H_

#include "ceres/codegen/codegen_cost_function.h"

namespace test {


// From the NIST problem collection.
struct Rat43CostFunctor : public ceres::CodegenCostFunction<1, 4> {
  Rat43CostFunctor() = default;
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* parameters, T* residuals) const {
    T b1 = parameters[0];
    T b2 = parameters[1];
    T b3 = parameters[2];
    T b4 = parameters[3];
    T x = CERES_LOCAL_VARIABLE(T,x_);
    T y = CERES_LOCAL_VARIABLE(T,y_);
    residuals[0] = b1 * pow(T(1.0) + exp(b2 - b3 * x), T(-1.0) / b4) - y;
    return true;
  }

#include "tests/rat43costfunctor.h"

 private:
  double x_ = 0;
  double y_ = 0;
};
}
#endif  // CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
