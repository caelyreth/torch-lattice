cmake_minimum_required(VERSION 3.28)

project(torch_lattice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENV{TORCH_CUDA_ARCH_LIST} "8.9")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  COMMAND_ERROR_IS_FATAL ANY
)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

option(TORCH_LATTICE_FORCE_CPU "Build TorchLattice without CUDA sources" OFF)

if(NOT TORCH_LATTICE_FORCE_CPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib', end='')"
  OUTPUT_VARIABLE TORCH_LIB_DIR
  COMMAND_ERROR_IS_FATAL ANY
)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_LIB_DIR}" REQUIRED NO_DEFAULT_PATH)

file(GLOB_RECURSE TORCH_LATTICE_CPU_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/torch_lattice/backend/*_cpu.cpp"
)
list(FILTER TORCH_LATTICE_CPU_SOURCES EXCLUDE REGEX "/pybind_[^/]+\\.(cpp|cu)$")

if(TORCH_LATTICE_FORCE_CPU)
  set(TORCH_LATTICE_PYBIND_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/torch_lattice/backend/pybind_cpu.cpp")
  set(TORCH_LATTICE_SOURCES ${TORCH_LATTICE_PYBIND_SOURCE} ${TORCH_LATTICE_CPU_SOURCES})
else()
  file(GLOB_RECURSE TORCH_LATTICE_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_lattice/backend/*_cuda.cu"
  )
  list(FILTER TORCH_LATTICE_CUDA_SOURCES EXCLUDE REGEX "/pybind_[^/]+\\.(cpp|cu)$")
  set(TORCH_LATTICE_PYBIND_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/torch_lattice/backend/pybind_cuda.cu")
  set(TORCH_LATTICE_SOURCES
    ${TORCH_LATTICE_PYBIND_SOURCE}
    ${TORCH_LATTICE_CPU_SOURCES}
    ${TORCH_LATTICE_CUDA_SOURCES}
  )
endif()

add_library(backend MODULE ${TORCH_LATTICE_SOURCES})

target_include_directories(backend PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/torch_lattice/backend"
  "${TORCH_INCLUDE_DIRS}"
  "${Python_INCLUDE_DIRS}"
)

target_link_libraries(backend PRIVATE
  "${TORCH_LIBRARIES}"
  "${TORCH_PYTHON_LIBRARY}"
  Python::Module
  OpenMP::OpenMP_CXX
)

target_compile_definitions(backend PRIVATE
  TORCH_API_INCLUDE_EXTENSION_H
  TORCH_EXTENSION_NAME=backend
)

target_compile_options(backend PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

if(NOT TORCH_LATTICE_FORCE_CPU)
  set_target_properties(backend PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "89"
  )
  target_compile_options(backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  )
  target_link_libraries(backend PRIVATE CUDA::cudart)
endif()

set_target_properties(backend PROPERTIES
  PREFIX ""
  OUTPUT_NAME "backend"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/torch_lattice"
)

install(TARGETS backend LIBRARY DESTINATION torch_lattice)
