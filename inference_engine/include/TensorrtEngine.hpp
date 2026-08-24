#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

// Tensorrt requires a logger implementation to get runtime data
class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* message) noexcept override;
};

struct TensorInfo {
  std::string name;
  nvinfer1::Dims dimensions;
  nvinfer1::DataType dataType;
  size_t sizeBytes;
  bool isInput;
};

// Wrapper for a single Tensorrt engine. Loads, allocates device buffers and
// runs inference. One instance corresponds to one engine
class TensorrtEngine {
public:
  explicit TensorrtEngine(const std::string &enginePath);
  ~TensorrtEngine();
  // Instance is not copyable
  TensorrtEngine(const TensorrtEngine &) = delete;
  TensorrtEngine &operator=(const TensorrtEngine &) = delete;
  // Runs inference with a single tensor of data, returns output tensor
  std::vector<std::vector<float>> infer(const std::vector<float> &inputData);
  const std::vector<TensorInfo> &getTensorInfos() const { return tensorInfos_; }

private:
  void loadEngine(const std::string &enginePath);
  void allocateBuffers();

  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<TensorInfo> tensorInfos_;
  std::vector<void *> deviceBuffers_;
  cudaStream_t stream_ = nullptr;
};
