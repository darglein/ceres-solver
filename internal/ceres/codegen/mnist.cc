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
// A simple example showing how to generate code for a cost functor

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "neural_functor.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// This file is generated with ceres::GenerateCodeForFunctor.
// http://ceres-solver.org/

void EvaluateResidual(double const* const* parameters, double* residuals) const
{
  double v_0;
  double v_1;
  double v_2;
  double v_3;
  double v_4;
  double v_5;
  double v_6;
  double v_7;
  double v_8;
  double v_9;
  double v_10;
  double v_11;
  double v_12;
  double v_13;
  double v_14;
  double v_15;
  v_0 = 1;
  v_1 = 0;
  v_2 = parameters[0][0];
  v_3 = parameters[0][1];
  v_4 = parameters[0][2];
  v_5 = v_4 * v_4;
  v_6 = v_3 * v_3;
  v_7 = v_2 * v_2;
  v_8 = v_6 + v_7;
  v_9 = v_5 + v_8;
  v_10 = std::sqrt(v_9);
  v_11 = v_0 / v_10;
  v_12 = v_3 * v_11;
  residuals[0] = v_12;
  residuals[1] = v_1;
  residuals[2] = v_1;
}

void EvaluateResidualAndJacobian(double const* const* parameters, double* residuals, double** jacobians) const
{
  double v_0;
  double v_1;
  double v_2;
  double v_3;
  double v_4;
  double v_5;
  double v_6;
  double v_7;
  double v_8;
  double v_9;
  double v_10;
  double v_11;
  double v_12;
  double v_13;
  double v_14;
  double v_15;
  double v_16;
  double v_17;
  double v_18;
  double v_19;
  double v_20;
  double v_21;
  double v_22;
  double v_23;
  double v_24;
  double v_25;
  double v_26;
  double v_27;
  double v_28;
  double v_29;
  double v_30;
  double v_31;
  double v_32;
  double v_33;
  double v_34;
  double v_35;
  double v_36;
  double v_37;
  double v_38;
  double v_39;
  double v_40;
  double v_41;
  double v_42;
  double v_43;
  double v_44;
  double v_45;
  double v_46;
  double v_47;
  double v_48;
  double v_49;
  double v_50;
  double v_51;
  double v_52;
  double v_53;
  double v_54;
  double v_55;
  v_0 = 0;
  v_1 = 1;
  v_2 = 2;
  v_3 = parameters[0][0];
  v_4 = parameters[0][1];
  v_5 = parameters[0][2];
  v_6 = v_5 * v_5;
  v_7 = v_5 + v_5;
  v_8 = v_4 * v_4;
  v_9 = v_4 + v_4;
  v_10 = v_3 * v_3;
  v_11 = v_3 + v_3;
  v_12 = v_8 + v_10;
  v_13 = v_6 + v_12;
  v_14 = std::sqrt(v_13);
  v_15 = v_2 * v_14;
  v_16 = v_1 / v_15;
  v_17 = v_11 * v_16;
  v_18 = v_9 * v_16;
  v_19 = v_7 * v_16;
  v_20 = v_1 / v_14;
  v_21 = v_17 * v_20;
  v_22 = -v_21;
  v_23 = v_20 * v_22;
  v_24 = v_18 * v_20;
  v_25 = -v_24;
  v_26 = v_20 * v_25;
  v_27 = v_19 * v_20;
  v_28 = -v_27;
  v_29 = v_20 * v_28;
  v_30 = v_4 * v_20;
  v_31 = v_4 * v_23;
  v_32 = v_4 * v_26;
  v_33 = v_20 + v_32;
  v_34 = v_4 * v_29;
  residuals[0] = v_30;
  residuals[1] = v_0;
  residuals[2] = v_0;
  jacobians[0][0] = v_31;
  jacobians[0][1] = v_33;
  jacobians[0][2] = v_34;
  jacobians[0][3] = v_0;
  jacobians[0][4] = v_0;
  jacobians[0][5] = v_0;
  jacobians[0][6] = v_0;
  jacobians[0][7] = v_0;
  jacobians[0][8] = v_0;
  jacobians[1][0] = v_0;
  jacobians[1][1] = v_0;
  jacobians[1][2] = v_0;
  jacobians[1][3] = v_0;
  jacobians[1][4] = v_0;
  jacobians[1][5] = v_0;
  jacobians[1][6] = v_0;
  jacobians[1][7] = v_0;
  jacobians[1][8] = v_0;
}

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const {
  if (!jacobians) {
    EvaluateResidual(parameters, residuals);
    return true;
    }
  double jacobians_data[18];
  double* jacobians_ptrs[] = {
      jacobians[0] ? jacobians[0] : jacobians_data,
      jacobians[1] ? jacobians[1] : jacobians_data,
      };
  EvaluateResidualAndJacobian(parameters, residuals, jacobians_ptrs);
  return true;
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;

  Problem problem;

  const double kTargetValue = 10.0;

  //  CostFunction* cost_function =
  //      new helloworld::HelloWorldCostFunction(kTargetValue);
  //  problem.AddResidualBlock(cost_function, NULL, &x);

  std::cout << test::nn_size << std::endl;

  CostFunction* cost_function =
      new AutoDiffCostFunction<test::NeuralBACost, 2, 6, 3>(
          new test::NeuralBACost);

  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  return 0;
}
