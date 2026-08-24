#include "TensorrtEngine.hpp"
#include "TensorrtUtils.hpp"

#include <NvInferRuntimeBase.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

// TrtLogger implementation
void TrtLogger::log(Severity severity, const char *message) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TRT] " << message << std::endl;
  }
}

// TensorrtEngine implementation
// Constructor and Destructor
TensorrtEngine::TensorrtEngine(const std::string &enginePath) {
  loadEngine(enginePath);
  allocateBuffers();

  if (cudaStreamCreate(&stream_) != cudaSuccess)
    throw std::runtime_error("Could not create CUDA stream");
}

TensorrtEngine::~TensorrtEngine() {
  for (auto *buffer : deviceBuffers_) {
    if (buffer)
      cudaFree(buffer);
  }
  if (stream_)
    cudaStreamDestroy(stream_);
}

// Method implementations
void TensorrtEngine::loadEngine(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Could not open engine file: " + enginePath);

  const size_t size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<char> engineData(size);
  if (!file.read(engineData.data(), static_cast<std::streamsize>(size)))
    throw std::runtime_error("Could not read engine file: " + enginePath);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_)
    throw std::runtime_error("Could not create runtime: " + enginePath);
  engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
  if (!engine_)
    throw std::runtime_error("Could not deserialize engine");
  context_.reset(engine_->createExecutionContext());
  if (!context_)
    throw std::runtime_error("Could not create execution context");
  std::cout << "Loaded engine: " << enginePath << std::endl;
}

void TensorrtEngine::allocateBuffers() {
  const int nbTensors = engine_->getNbIOTensors();
  deviceBuffers_.resize(static_cast<size_t>(nbTensors), nullptr);

  for (int i = 0; i < nbTensors; ++i) {
    const char *name = engine_->getIOTensorName(i);
    const nvinfer1::Dims dimensions = engine_->getTensorShape(name);
    const nvinfer1::DataType dataType = engine_->getTensorDataType(name);
    const bool isInput =
        engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    const size_t bytes = volume(dimensions) * dTypeSize(dataType);
    void *devPtr = nullptr;
    if (cudaMalloc(&devPtr, bytes) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed for tensor " +
                               std::string(name) + " (" +
                               std::to_string(bytes) + " bytes)");
    }
    deviceBuffers_[static_cast<size_t>(i)] = devPtr;
    context_->setTensorAddress(name, devPtr);
    tensorInfos_.push_back(
        TensorInfo{name, dimensions, dataType, bytes, isInput});
    std::cout << (isInput ? "Input  " : "Output ") << name << " shape=[";
    for (int d = 0; d < dimensions.nbDims; ++d) {
      std::cout << dimensions.d[d] << (d + 1 < dimensions.nbDims ? "," : "");
    }
    std::cout << "] bytes=" << bytes << std::endl;
  }
}
