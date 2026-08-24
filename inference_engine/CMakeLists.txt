cmake_minimum_required(VERSION 3.14)
project(edgesched_inference LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu CACHE PATH "TensorRT include dir")
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu CACHE PATH "TensorRT lib dir")

include_directories(
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
link_directories(${TENSORRT_LIB_DIR})

add_executable(test_engine
    src/TensorrtEngine.cpp
    src/main.cpp
)

target_link_libraries(test_engine
    CUDA::cudart
    nvinfer
)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_utils
    tests/test_utils.cpp
)

target_link_libraries(test_utils
    GTest::gtest_main
    CUDA::cudart
)

include(GoogleTest)
gtest_discover_tests(test_utils)
