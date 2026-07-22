cmake_minimum_required(VERSION 3.15)
project(XeCUDA VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Source files — all translation units
set(XECUDA_SOURCES
    src/xecuda_runtime.cpp
    src/xecuda_blas.cpp
    src/xecuda_xmx.cpp
    src/xecuda_l0.cpp
    src/xecuda_stream.cpp
)

# OpenMP for parallel multi-threading emulation
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling Xe Core parallel SIMD execution.")
endif()

# Core Library
add_library(xecuda STATIC ${XECUDA_SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(xecuda PRIVATE OpenMP::OpenMP_CXX)
endif()

# Benchmark & Diagnostic CLI Executable
add_executable(xecuda-bench src/main_bench.cpp)
target_link_libraries(xecuda-bench PRIVATE xecuda)

if(OpenMP_CXX_FOUND)
    target_link_libraries(xecuda-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

message(STATUS "XeCUDA project configured for Intel Arc GPUs!")
