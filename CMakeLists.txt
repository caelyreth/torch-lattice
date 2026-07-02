cmake_minimum_required(VERSION 3.28)

project(torchsparse LANGUAGES CXX)

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

option(TORCHSPARSE_FORCE_CPU "Build TorchSparse without CUDA sources" OFF)

if(NOT TORCHSPARSE_FORCE_CPU)
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

file(GLOB_RECURSE TORCHSPARSE_CPU_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/torchsparse/backend/*_cpu.cpp"
)
list(FILTER TORCHSPARSE_CPU_SOURCES EXCLUDE REGEX "/pybind_[^/]+\\.(cpp|cu)$")

if(TORCHSPARSE_FORCE_CPU)
  set(TORCHSPARSE_PYBIND_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/torchsparse/backend/pybind_cpu.cpp")
  set(TORCHSPARSE_SOURCES ${TORCHSPARSE_PYBIND_SOURCE} ${TORCHSPARSE_CPU_SOURCES})
else()
  file(GLOB_RECURSE TORCHSPARSE_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/torchsparse/backend/*_cuda.cu"
  )
  list(FILTER TORCHSPARSE_CUDA_SOURCES EXCLUDE REGEX "/pybind_[^/]+\\.(cpp|cu)$")
  set(TORCHSPARSE_PYBIND_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/torchsparse/backend/pybind_cuda.cu")
  set(TORCHSPARSE_SOURCES
    ${TORCHSPARSE_PYBIND_SOURCE}
    ${TORCHSPARSE_CPU_SOURCES}
    ${TORCHSPARSE_CUDA_SOURCES}
  )
endif()

add_library(backend MODULE ${TORCHSPARSE_SOURCES})

target_include_directories(backend PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/torchsparse/backend"
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

if(NOT TORCHSPARSE_FORCE_CPU)
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
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/torchsparse"
)

install(TARGETS backend LIBRARY DESTINATION torchsparse)
