cmake_minimum_required(VERSION 3.14)
project(edgesched_inference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu CACHE PATH "TensorRT include dir")
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu CACHE PATH "TensorRT lib dir")
include_directories(
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
link_directories(${TENSORRT_LIB_DIR})
find_package(OpenCV REQUIRED)

# Protobuf + gRPC
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
set(PROTO_DIR ${CMAKE_SOURCE_DIR}/../proto)
set(PROTO_FILE ${PROTO_DIR}/inference.proto)
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})
set(PROTO_SRCS ${GENERATED_DIR}/inference.pb.cc)
set(PROTO_HDRS ${GENERATED_DIR}/inference.pb.h)
set(GRPC_SRCS ${GENERATED_DIR}/inference.grpc.pb.cc)
set(GRPC_HDRS ${GENERATED_DIR}/inference.grpc.pb.h)
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --grpc_out=${GENERATED_DIR}
         --cpp_out=${GENERATED_DIR}
         -I ${PROTO_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC/protobuf stubs from inference.proto"
)
add_library(proto_generated STATIC ${PROTO_SRCS} ${GRPC_SRCS})
target_include_directories(proto_generated PUBLIC ${GENERATED_DIR})
target_link_libraries(proto_generated PUBLIC protobuf::libprotobuf PkgConfig::GRPCPP)

# Targets
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
add_executable(test_pre_postprocess
    src/Preprocess.cpp
    src/Postprocess.cpp
    tests/test_pre_postprocess.cpp
)
target_include_directories(test_pre_postprocess PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_pre_postprocess
    GTest::gtest_main
    ${OpenCV_LIBS}
)
gtest_discover_tests(test_pre_postprocess)

# Inference server
add_executable(inference_server
    src/TensorrtEngine.cpp
    src/Preprocess.cpp
    src/Postprocess.cpp
    src/InferenceServiceImpl.cpp
    src/server_main.cpp
)
target_include_directories(inference_server PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference_server
    proto_generated
    CUDA::cudart
    nvinfer
    ${OpenCV_LIBS}
)

# Test client
add_executable(test_client
    client/test_client.cpp
)
target_include_directories(test_client PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_client
    proto_generated
    ${OpenCV_LIBS}
)
