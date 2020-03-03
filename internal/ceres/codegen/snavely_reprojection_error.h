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

template <typename T>
inline void AngleAxisRotatePoint2(const T angle_axis[3], T result[3]) {
#if 0
  T v_0 = T(181778);
  T v_1 = T(3755064);
  T v_2 = T(937);
  T v_3 = T(19356);
  T v_4 = T(194);
  T v_5 = angle_axis[0];
  T v_6 = v_2 * v_5;
  T v_7 = v_1 * v_6;
  T v_8 = v_0 * v_6;
  T v_9 = v_5 * v_7;
  T v_10 = v_4 * v_5;
  T v_11 = v_7 + v_10;
  T v_12 = v_6 * v_8;
  T v_13 = v_2 * v_8;
  T v_14 = v_3 * v_6;
  T v_15 = v_13 + v_14;
  T v_16 = v_9 + v_12;
  T v_17 = v_11 + v_15;
  result[0] = v_16;
  result[1] = v_17;
#else
  T theta2 = angle_axis[0] * T(194);

  const T theta = theta2 * T(19356);
  const T sintheta = theta2 * T(937);

  result[0] = theta2 * sintheta + angle_axis[0] * theta;
//  result[1] = theta2 * sintheta + angle_axis[0] * theta;
#endif
}

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct RotatePoint2 : public ceres::CodegenCostFunction<1, 1> {
  template <typename T>
  bool operator()(const T* const angle_axis, T* result) const {
    //    ceres::AngleAxisRotatePoint(angle_axis, pt, result);
    AngleAxisRotatePoint2(angle_axis, result);
    return true;
  }

#include "tests/rotatepoint.h"
};

struct RotatePoint : public ceres::CodegenCostFunction<2, 8> {
  template <typename T>
  bool operator()(const T* const x, T* y) const {
    T c1(2);
    T c2(3);
    T c3(4);

    T theta2 = x[0] * c1;
    T theta = theta2 * c2;
    T sintheta = theta2 * c3;

    y[0] = theta2 * sintheta + x[0] * theta;
    y[1] = T(0);
    //    y[0] = x[2] * x[1] * x[0];
    //    y[1] = x[0] * x[1] * x[2];

    return true;
  }

#include "tests/rotatepoint.h"
};

struct SnavelyReprojectionErrorGen
    : public ceres::CodegenCostFunction<2, 9, 3> {
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

    //    residuals[0] = ox - p[0];
    //    residuals[1] = T(0);
    //    residuals[1] = ox - p[1];
    //    return true;

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

}  // namespace test
#endif  // CERES_EXAMPLES_SNAVELY_REPROJECTION_ERROR_H_
