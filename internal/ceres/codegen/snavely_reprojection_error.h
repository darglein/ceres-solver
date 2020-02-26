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

#ifndef CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR2_H_
#define CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR2_H_

#include "ceres/codegen/codegen_cost_function.h"
#include "ceres/rotation.h"
#include "common.h"

namespace test {

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct RotatePoint : public ceres::CodegenCostFunction<3, 3, 3> {
  RotatePoint() = default;

  template <typename T>
  bool operator()(const T* const angle_axis,
                  const T* const pt,
                  T* result) const {
    ceres::AngleAxisRotatePoint(angle_axis, pt, result);
    return true;
  }

  //#include "tests/rotatepoint.h"
};

using RotatePointAD = ceres::internal::CostFunctionToFunctor<RotatePoint>;

struct SnavelyReprojectionErrorGen
    : public ceres::CodegenCostFunction<2, 6, 3> {
  SnavelyReprojectionErrorGen(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  SnavelyReprojectionErrorGen() = default;
  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
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

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    const T r2 = xp * xp + yp * yp;
    const T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - ox;
    residuals[1] = predicted_y - oy;

    return true;
  }

#include "tests/snavelyreprojectionerrorgen.h"
  double observed_x;
  double observed_y;
};

using SnavelyReprojectionAD =
    ceres::internal::CostFunctionToFunctor<SnavelyReprojectionErrorGen>;

}  // namespace test
#endif  // CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_