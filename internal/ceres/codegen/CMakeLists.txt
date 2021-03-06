# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2019 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: darius.rueckert@fau.de (Darius Rueckert)

if (BUILD_TESTING AND GFLAGS)
  # Testing the AutoDiffCodegen system is more complicated, because function- and
  # constructor calls have side-effects. In C++ the evaluation order and
  # the elision of copies is implementation defined. Between different compilers,
  # some expression might be evaluated in a different order or some copies might be
  # removed.
  #
  # Therefore, we run tests that expect a particular compiler behaviour only on gcc.
  #
  # The semantic tests, which check the correctness by executing the generated code
  # are still run on all platforms.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CXX_FLAGS_OLD ${CMAKE_CXX_FLAGS})
    # From the man page:
    #   The C++ standard allows an implementation to omit creating a
    #   temporary which is only used to initialize another object of the
    #   same type.  Specifying -fno-elide-constructors disables that
    #   optimization, and forces G++ to call the copy constructor in all cases.
    # We use this flag to get the same results between different versions of
    # gcc and different optimization levels. Without this flag, testing would
    # be very painfull and might break when a new compiler version is released.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-elide-constructors")
    ceres_test(expression_ref)
    ceres_test(code_generator)
    ceres_test(cfg)
    ceres_test(eliminate_nops)
    ceres_test(trivial_assignment)
    set(CMAKE_CXX_FLAGS ${CXX_FLAGS_OLD})
  endif()


  ceres_test(expression)
  ceres_test(expression_graph)
  ceres_test(expression_dependencies)
  ceres_test(merge_constants)

  macro (generate_test_functor FUNCTOR_NAME FUNCTOR_FILE)
    ceres_generate_cost_function_implementation_for_functor(
      NAME ${FUNCTOR_NAME}
      INPUT_FILE ${FUNCTOR_FILE}
      OUTPUT_DIRECTORY tests
      NAMESPACE test
      )
  endmacro()

  # Semantic tests should work on every platform
  include(CeresCodeGeneration)


  #  generate_test_functor(MnistFunctor neural_functor.h)
  #  add_executable(mnist mnist.cc )
  #  target_link_libraries(mnist ceres)

  generate_test_functor(InputOutputAssignment autodiff_codegen_test.h)
  generate_test_functor(CompileTimeConstants autodiff_codegen_test.h)
  generate_test_functor(Assignments autodiff_codegen_test.h)
  generate_test_functor(BinaryArithmetic autodiff_codegen_test.h)
  generate_test_functor(UnaryArithmetic autodiff_codegen_test.h)
  generate_test_functor(BinaryComparison autodiff_codegen_test.h)
  generate_test_functor(LogicalOperators autodiff_codegen_test.h)
  generate_test_functor(ScalarFunctions autodiff_codegen_test.h)
  generate_test_functor(LogicalFunctions autodiff_codegen_test.h)
  generate_test_functor(Branches autodiff_codegen_test.h)


  ceres_test(to_partial_ssa Branches)
  ceres_test(reorder)


  ceres_test(autodiff_codegen
    InputOutputAssignment CompileTimeConstants Assignments BinaryArithmetic
    UnaryArithmetic BinaryComparison LogicalOperators ScalarFunctions
    LogicalFunctions Branches
    )

  generate_test_functor(DisneyBRDF brdf_cost_function.h)
  ceres_test(brdf DisneyBRDF)

  generate_test_functor(Projective gen_autodiff_test_functors.h)
  generate_test_functor(Residual1Param gen_autodiff_test_functors.h)
  generate_test_functor(Residual2Param gen_autodiff_test_functors.h)
  generate_test_functor(Residual3Param gen_autodiff_test_functors.h)
  generate_test_functor(Residual4Param gen_autodiff_test_functors.h)
  ceres_test(gen_autodiff Projective Residual1Param Residual2Param Residual3Param Residual4Param)

  generate_test_functor(SnavelyReprojectionErrorGen snavely_reprojection_error.h)
  ceres_test(gen_reprojection_error SnavelyReprojectionErrorGen)
endif()

if (BUILD_BENCHMARKS)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND CERES_BENCHMARK_FLAGS "-march=native")
    list(APPEND CERES_BENCHMARK_FLAGS "-fno-math-errno" "-fno-trapping-math" "-fassociative-math" "-fno-signed-zeros" "-ffp-contract=fast")
    list(APPEND CERES_BENCHMARK_FLAGS "-ffinite-math-only")
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND CERES_BENCHMARK_FLAGS "-march=native")
    list(APPEND CERES_BENCHMARK_FLAGS "-fno-math-errno" "-fno-trapping-math" "-fassociative-math" "-fno-signed-zeros" "-ffp-contract=fast")
    list(APPEND CERES_BENCHMARK_FLAGS "-fno-honor-infinities" "-fno-honor-nans")
    list(APPEND CERES_BENCHMARK_FLAGS "-mllvm" "-inline-threshold=6000000")

  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    list(APPEND CERES_BENCHMARK_FLAGS "-march=native")
    list(APPEND CERES_BENCHMARK_FLAGS "-fno-math-errno" "-fno-trapping-math" "-fassociative-math" "-fno-signed-zeros" "-ffp-contract=fast")
    list(APPEND CERES_BENCHMARK_FLAGS "-fno-honor-infinities" "-fno-honor-nans")
  endif()

  generate_test_functor(Rat43CostFunctor rat43.h)
  add_executable(gen_rat43_benchmark gen_rat43_benchmark.cc)
  add_dependencies_to_benchmark(gen_rat43_benchmark Rat43CostFunctor)

  add_executable(gen_ba_benchmark gen_ba_benchmark.cc)
  add_dependencies_to_benchmark(gen_ba_benchmark SnavelyReprojectionErrorGen)

  add_executable(gen_ba_benchmark_fast_math gen_ba_benchmark.cc)
  add_dependencies_to_benchmark(gen_ba_benchmark_fast_math SnavelyReprojectionErrorGen)
  target_compile_options(gen_ba_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FLAGS})

  add_executable(brdf_benchmark brdf_benchmark.cc)
  add_dependencies_to_benchmark(brdf_benchmark DisneyBRDF)

  add_executable(brdf_benchmark_fast_math brdf_benchmark.cc)
  add_dependencies_to_benchmark(brdf_benchmark_fast_math DisneyBRDF)
  target_compile_options(brdf_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FLAGS})

  #  target_compile_options(brdf_benchmark_fast_math PRIVATE -fno-math-errno)
  #  target_compile_options(brdf_benchmark_fast_math PRIVATE -fno-honor-nans)
  #  target_compile_options(brdf_benchmark_fast_math PRIVATE -fassociative-math)
  #  target_compile_options(brdf_benchmark_fast_math PRIVATE -fno-signed-zeros)

  #  target_compile_options(brdf_benchmark_fast_math PRIVATE -fno-honor-infinities)
  #  target_compile_options(brdf_benchmark_fast_math PRIVATE -fno-trapping-math)

  generate_test_functor(NeuralBACost network_cost_function.h)

  add_executable(network_benchmark network_benchmark.cc)
  add_dependencies_to_benchmark(network_benchmark ceres NeuralBACost)

  add_executable(network_benchmark_fast_math network_benchmark.cc)
  add_dependencies_to_benchmark(network_benchmark_fast_math ceres NeuralBACost)
  target_compile_options(network_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FLAGS})

endif()
