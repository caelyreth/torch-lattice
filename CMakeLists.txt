cmake_minimum_required(VERSION 3.28)

project(torch_lattice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  COMMAND_ERROR_IS_FATAL ANY
)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

option(TORCH_LATTICE_FORCE_CPU "Build TorchLattice without CUDA sources" OFF)

# Torch configures CUDA code generation through TORCH_CUDA_ARCH_LIST and
# explicitly ignores CMAKE_CUDA_ARCHITECTURES. Keep one canonical architecture
# setting so every translation unit is compiled for exactly the requested set.
if(NOT DEFINED TORCH_CUDA_ARCH_LIST AND NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  # RTX 2080, A100, RTX 4090, H100/H20, and RTX 5090 respectively.
  set(
    TORCH_CUDA_ARCH_LIST
    "7.5;8.0;8.9;9.0;12.0"
    CACHE STRING
    "CUDA architectures included in torch-lattice wheels"
  )
endif()

if(NOT TORCH_LATTICE_FORCE_CPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # CMake initializes this cache entry from the compiler's host default, but
  # PyTorch ignores it in favor of TORCH_CUDA_ARCH_LIST. Remove the competing
  # setting before loading Torch so headless builders cannot select extra SMs.
  unset(CMAKE_CUDA_ARCHITECTURES)
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
endif()

find_package(Torch REQUIRED)

if(NOT TORCH_LATTICE_FORCE_CPU)
  # Torch has now translated TORCH_CUDA_ARCH_LIST into NVCC flags. Mark CMake's
  # parallel architecture mechanism as intentionally disabled for our target.
  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib', end='')"
  OUTPUT_VARIABLE TORCH_LIB_DIR
  COMMAND_ERROR_IS_FATAL ANY
)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_LIB_DIR}" REQUIRED NO_DEFAULT_PATH)
find_file(
  TORCH_OPENMP_LIBRARY
  NAMES libgomp.so.1
  PATHS "${TORCH_LIB_DIR}"
  REQUIRED
  NO_DEFAULT_PATH
)

file(GLOB_RECURSE TORCH_LATTICE_CPU_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/native/*_cpu.cpp"
)
list(FILTER TORCH_LATTICE_CPU_SOURCES EXCLUDE REGEX "/pybind_[^/]+\\.(cpp|cu)$")

if(TORCH_LATTICE_FORCE_CPU)
  set(TORCH_LATTICE_PYBIND_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/native/pybind_cpu.cpp")
  set(TORCH_LATTICE_SOURCES ${TORCH_LATTICE_PYBIND_SOURCE} ${TORCH_LATTICE_CPU_SOURCES})
else()
  file(GLOB_RECURSE TORCH_LATTICE_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/native/*_cuda.cu"
  )
  list(FILTER TORCH_LATTICE_CUDA_SOURCES EXCLUDE REGEX "/pybind_[^/]+\\.(cpp|cu)$")
  set(TORCH_LATTICE_PYBIND_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/native/pybind_cuda.cu")
  set(TORCH_LATTICE_SOURCES
    ${TORCH_LATTICE_PYBIND_SOURCE}
    ${TORCH_LATTICE_CPU_SOURCES}
    ${TORCH_LATTICE_CUDA_SOURCES}
  )
endif()

add_library(torch_lattice_native MODULE ${TORCH_LATTICE_SOURCES})

if(NOT TORCH_LATTICE_FORCE_CPU)
  # These diagnostics come from compile-time variants in the inherited
  # TorchSparse convolution kernels. Restrict suppression to those sources so
  # warnings in maintained bindings and newer kernels remain actionable.
  set(TORCH_LATTICE_LEGACY_CONVOLUTION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/native/convolution/convolution_backward_wgrad_implicit_gemm_cuda.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/convolution/convolution_backward_wgrad_implicit_gemm_sorted_cuda.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/convolution/convolution_forward_fetch_on_demand_cuda.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/convolution/convolution_forward_implicit_gemm_cuda.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/convolution/convolution_forward_implicit_gemm_sorted_cuda.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/convolution/convolution_gather_scatter_cuda.cu"
  )
  set_source_files_properties(
    ${TORCH_LATTICE_LEGACY_CONVOLUTION_SOURCES}
    PROPERTIES COMPILE_OPTIONS "--diag-suppress=177;--diag-suppress=550"
  )
endif()

target_include_directories(torch_lattice_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/native"
  "${TORCH_INCLUDE_DIRS}"
  "${Python_INCLUDE_DIRS}"
)

target_link_libraries(torch_lattice_native PRIVATE
  "${TORCH_LIBRARIES}"
  "${TORCH_PYTHON_LIBRARY}"
  "${TORCH_OPENMP_LIBRARY}"
  Python::Module
)

target_compile_definitions(torch_lattice_native PRIVATE
  TORCH_API_INCLUDE_EXTENSION_H
  TORCH_EXTENSION_NAME=_C
)

target_compile_options(torch_lattice_native PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

if(NOT TORCH_LATTICE_FORCE_CPU)
  set_target_properties(torch_lattice_native PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(torch_lattice_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  )
  target_link_libraries(torch_lattice_native PRIVATE CUDA::cudart)
endif()

set_target_properties(torch_lattice_native PROPERTIES
  PREFIX ""
  OUTPUT_NAME "_C"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/torch_lattice"
)

install(TARGETS torch_lattice_native LIBRARY DESTINATION torch_lattice)
