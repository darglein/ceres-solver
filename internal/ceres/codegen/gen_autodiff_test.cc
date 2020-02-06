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
#include "ceres/codegen/internal/expression.h"
#include "ceres/internal/autodiff.h"
#include "ceres/random.h"
#include "gen_autodiff_test_functors.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Do (symmetric) finite differencing using the given function object 'b' of
// type 'B' and scalar type 'T' with step size 'del'.
//
// The type B should have a signature
//
//   bool operator()(T const *, T *) const;
//
// which maps a vector of parameters to a vector of outputs.
template <typename B, typename T, int M, int N>
inline bool SymmetricDiff(const B& b,
                          const T par[N],
                          T del,  // step size.
                          T fun[M],
                          T jac[M * N]) {  // row-major.
  if (!b(par, fun)) {
    return false;
  }

  // Temporary parameter vector.
  T tmp_par[N];
  for (int j = 0; j < N; ++j) {
    tmp_par[j] = par[j];
  }

  // For each dimension, we do one forward step and one backward step in
  // parameter space, and store the output vector vectors in these vectors.
  T fwd_fun[M];
  T bwd_fun[M];

  for (int j = 0; j < N; ++j) {
    // Forward step.
    tmp_par[j] = par[j] + del;
    if (!b(tmp_par, fwd_fun)) {
      return false;
    }

    // Backward step.
    tmp_par[j] = par[j] - del;
    if (!b(tmp_par, bwd_fun)) {
      return false;
    }

    // Symmetric differencing:
    //   f'(a) = (f(a + h) - f(a - h)) / (2 h)
    for (int i = 0; i < M; ++i) {
      test::RowMajorAccess(jac, M, N, i, j) =
          (fwd_fun[i] - bwd_fun[i]) / (T(2) * del);
    }

    // Restore our temporary vector.
    tmp_par[j] = par[j];
  }

  return true;
}

// Test projective camera model projector.
TEST(AutoDiff, ProjectiveCameraModel) {
  srand(5);
  double const tol = 1e-10;  // floating-point tolerance.
  double const del = 1e-4;   // finite-difference step.
  double const err = 1e-6;   // finite-difference tolerance.

  test::Projective b;

  // Make random P and X, in a single vector.
  double PX[12 + 4];
  for (int i = 0; i < 12 + 4; ++i) {
    PX[i] = RandDouble();
  }

  // Handy names for the P and X parts.
  double* P = PX + 0;
  double* X = PX + 12;

  // Apply the mapping, to get image point b_x.
  double b_x[2];
  b(P, X, b_x);

  // Use finite differencing to estimate the Jacobian.
  double fd_x[2];
  double fd_J[2 * (12 + 4)];
  ASSERT_TRUE((SymmetricDiff<test::Projective, double, 2, 12 + 4>(
      b, PX, del, fd_x, fd_J)));

  for (int i = 0; i < 2; ++i) {
    ASSERT_NEAR(fd_x[i], b_x[i], tol);
  }

  // Use automatic differentiation to compute the Jacobian.
  double ad_x1[2];
  double J_PX[2 * (12 + 4)];
  {
    double* parameters[] = {P, X};
    double* jacobians[] = {J_PX};
    ASSERT_TRUE(b.Evaluate(parameters, ad_x1, jacobians));

    for (int i = 0; i < 2; ++i) {
      ASSERT_NEAR(ad_x1[i], b_x[i], tol);
    }
  }
}

TEST(CodegenAutoDiff, VariadicAutoDiff) {
  double x[10];
  double residual = 0;
  double* parameters[10];
  double jacobian_values[10];
  double* jacobians[10];

  for (int i = 0; i < 10; ++i) {
    x[i] = 2.0;
    parameters[i] = x + i;
    jacobians[i] = jacobian_values + i;
  }

  {
    test::Residual1Param functor;
    int num_variables = 1;
    EXPECT_TRUE(functor.Evaluate(parameters, &residual, jacobians));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    test::Residual2Param functor;
    int num_variables = 2;
    EXPECT_TRUE(functor.Evaluate(parameters, &residual, jacobians));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    test::Residual3Param functor;
    int num_variables = 3;
    EXPECT_TRUE(functor.Evaluate(parameters, &residual, jacobians));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    test::Residual4Param functor;
    int num_variables = 4;
    EXPECT_TRUE(functor.Evaluate(parameters, &residual, jacobians));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }
}

}  // namespace internal
}  // namespace ceres
