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
#ifndef CERES_PUBLIC_CODEGEN_AUTODIFF_H_
#define CERES_PUBLIC_CODEGEN_AUTODIFF_H_

#include "ceres/codegen/internal/code_generator.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/optimize_expression_graph.h"
#include "ceres/codegen/internal/trace.h"

namespace ceres {

struct AutoDiffCodeGenOptions {};

// TODO(darius): Documentation
template <typename DerivedCostFunctor>
std::vector<std::string> GenerateCodeForFunctor(
    const AutoDiffCodeGenOptions& options) {
  // Define some types and shortcuts to make the code below more readable.
  using ParameterDims = typename DerivedCostFunctor::ParameterDims;
  using Parameters = typename ParameterDims::Parameters;
  constexpr int kNumResiduals = DerivedCostFunctor::kNumResiduals;
  constexpr int kNumParameters = ParameterDims::kNumParameters;
  constexpr int kNumParameterBlocks = ParameterDims::kNumParameterBlocks;

  auto residual_graph = internal::TraceFunctor<DerivedCostFunctor>();

  auto residual_and_jacobian_graph =
      internal::TraceFunctorDerivative<DerivedCostFunctor>();

  // TODO(darius): Once the optimizer is in place, call it from
  // here to optimize the code before generating.
  //  internal::OptimizeExpressionGraphOptions optimization_options;
  //  auto summary_residual =
  //      internal::SuperOptimize(optimization_options, &residual_graph);
  //  auto summary_jacobian = internal::SuperOptimize(optimization_options,
  //                                                  &residual_and_jacobian_graph);
  //  std::cout << summary_jacobian << std::endl;

  // We have the optimized code of the cost functor stored in the
  // ExpressionGraphs. Now we generate C++ code for it and place it line-by-line
  // in this vector of strings.
  std::vector<std::string> output;

  output.emplace_back(
      "// This file is generated with ceres::GenerateCodeForFunctor.");
  output.emplace_back("// http://ceres-solver.org/");
  output.emplace_back("");

  {
    // Generate C++ code for the EvaluateResidual function and append it to the
    // output.
    internal::CodeGenerator::Options generator_options;
    generator_options.function_name =
        "void EvaluateResidual(double const* const* parameters, double* "
        "residuals) const";
    internal::CodeGenerator gen(residual_graph, generator_options);
    std::vector<std::string> code = gen.Generate();
    output.insert(output.end(), code.begin(), code.end());
  }

  output.emplace_back("");

  {
    // Generate C++ code for the EvaluateResidualAndJacobian function and append
    // it to the output.
    internal::CodeGenerator::Options generator_options;
    generator_options.function_name =
        "void EvaluateResidualAndJacobian(double const* const* parameters, "
        "double* "
        "residuals, double** jacobians) const";
    internal::CodeGenerator gen(residual_and_jacobian_graph, generator_options);
    std::vector<std::string> code = gen.Generate();
    output.insert(output.end(), code.begin(), code.end());
  }

  output.emplace_back("");

  // Generate a generic combined function, which calls EvaluateResidual and
  // EvaluateResidualAndJacobian. This combined function is compatible to
  // CostFunction::Evaluate. Therefore the generated code can be directly used
  // in SizedCostFunctions.
  output.emplace_back("bool Evaluate(double const* const* parameters,");
  output.emplace_back("              double* residuals,");
  output.emplace_back("              double** jacobians) const {");

  output.emplace_back("   if (!jacobians) {");
  output.emplace_back("     EvaluateResidual(parameters, residuals);");
  output.emplace_back("     return true;");
  output.emplace_back("   }");

  // Create a tmp array of all jacobians and use it for evaluation.
  // The generated code for a <2,3,1,2> cost functor is:
  //   double jacobians_data[6];
  //   double* jacobians_ptrs[] = {
  //       jacobians_data + 0,
  //       jacobians_data + 6,
  //       jacobians_data + 8,
  //   };
  output.emplace_back("   double jacobians_data[" +
                      std::to_string(kNumParameters * kNumResiduals) + "];");
  output.emplace_back("   double* jacobians_ptrs[] = {");
  for (int i = 0, total_param_id = 0; i < kNumParameterBlocks;
       total_param_id += ParameterDims::GetDim(i), ++i) {
    output.emplace_back("     jacobians[" + std::to_string(i) +
                        "] ? jacobians[" + std::to_string(i) +
                        "] : jacobians_data,");
  }
  output.emplace_back("   };");

  // Evaluate into the tmp array.
  output.emplace_back(
      "   EvaluateResidualAndJacobian(parameters, residuals, "
      "jacobians_ptrs);");

  output.emplace_back("   return true;");
  output.emplace_back("}");

  return output;
}

}  // namespace ceres
#endif  // CERES_PUBLIC_CODEGEN_AUTODIFF_H_
