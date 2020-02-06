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

namespace test {

template <typename T>
inline T& RowMajorAccess(T* base, int rows, int cols, int i, int j) {
  return base[cols * i + j];
}

template <typename A>
inline void QuaternionToScaledRotation(A const q[4], A R[3 * 3]) {
  // Make convenient names for elements of q.
  A a = q[0];
  A b = q[1];
  A c = q[2];
  A d = q[3];
  // This is not to eliminate common sub-expression, but to
  // make the lines shorter so that they fit in 80 columns!
  A aa = a * a;
  A ab = a * b;
  A ac = a * c;
  A ad = a * d;
  A bb = b * b;
  A bc = b * c;
  A bd = b * d;
  A cc = c * c;
  A cd = c * d;
  A dd = d * d;
#define R(i, j) RowMajorAccess(R, 3, 3, (i), (j))
  R(0, 0) = aa + bb - cc - dd;
  R(0, 1) = A(2) * (bc - ad);
  R(0, 2) = A(2) * (ac + bd);  // NOLINT
  R(1, 0) = A(2) * (ad + bc);
  R(1, 1) = aa - bb + cc - dd;
  R(1, 2) = A(2) * (cd - ab);  // NOLINT
  R(2, 0) = A(2) * (bd - ac);
  R(2, 1) = A(2) * (ab + cd);
  R(2, 2) = aa - bb - cc + dd;  // NOLINT
#undef R
}

// A structure for projecting a 3x4 camera matrix and a
// homogeneous 3D point, to a 2D inhomogeneous point.
struct Projective : public ceres::CodegenCostFunction<2, 12, 4> {
  // Function that takes P and X as separate vectors:
  //   P, X -> x
  template <typename A>
  bool operator()(A const P[12], A const X[4], A x[2]) const {
    A PX[3];
    for (int i = 0; i < 3; ++i) {
      PX[i] = RowMajorAccess(P, 3, 4, i, 0) * X[0] +
              RowMajorAccess(P, 3, 4, i, 1) * X[1] +
              RowMajorAccess(P, 3, 4, i, 2) * X[2] +
              RowMajorAccess(P, 3, 4, i, 3) * X[3];
    }

    x[0] = PX[0] / PX[2];
    x[1] = PX[1] / PX[2];
    return true;
  }

  // Version that takes P and X packed in one vector:
  //
  //   (P, X) -> x
  //
  template <typename A>
  bool operator()(A const P_X[12 + 4], A x[2]) const {
    return operator()(P_X + 0, P_X + 12, x);
  }
#include "tests/projective.h"
};

struct Residual1Param : public ceres::CodegenCostFunction<1, 1> {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    y[0] = *x0;
    return true;
  }
#include "tests/residual1param.h"
};

struct Residual2Param : public ceres::CodegenCostFunction<1, 1, 1> {
  template <typename T>
  bool operator()(const T* x0, const T* x1, T* y) const {
    y[0] = *x0 + pow(*x1, T(2));
    return true;
  }
#include "tests/residual2param.h"
};

struct Residual3Param : public ceres::CodegenCostFunction<1, 1, 1, 1> {
  template <typename T>
  bool operator()(const T* x0, const T* x1, const T* x2, T* y) const {
    y[0] = *x0 + pow(*x1, T(2)) + pow(*x2, T(3));
    return true;
  }
#include "tests/residual3param.h"
};

struct Residual4Param : public ceres::CodegenCostFunction<1, 1, 1, 1, 1> {
  template <typename T>
  bool operator()(
      const T* x0, const T* x1, const T* x2, const T* x3, T* y) const {
    y[0] = *x0 + pow(*x1, T(2)) + pow(*x2, T(3)) + pow(*x3, T(4));
    return true;
  }
#include "tests/residual4param.h"
};

}  // namespace test
