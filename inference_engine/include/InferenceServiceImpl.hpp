#pragma once

#include "Postprocess.hpp"
#include "Preprocess.hpp"
#include "TensorrtEngine.hpp"
#include "inference.grpc.pb.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Implements the Inference gRPC service (Predict, HealthCheck) declared in
// proto/inference.proto, backed by a single TensorRTEngine. One instance
// of this class = one engine = one gRPC server process
class InferenceServiceImpl final : public edgesched::Inference::Service {
public:
  InferenceServiceImpl(std::shared_ptr<TensorrtEngine> engine,
                       std::string engineName,
                       std::vector<std::string> classNames,
                       float confThreshold = 0.25f, int inputSize = 640);

  grpc::Status Predict(grpc::ServerContext *context,
                       const edgesched::PredictRequest *request,
                       edgesched::PredictResponse *response) override;

  grpc::Status HealthCheck(grpc::ServerContext *context,
                           const edgesched::HealthCheckRequest *request,
                           edgesched::HealthCheckResponse *response) override;

private:
  std::shared_ptr<TensorrtEngine> engine_;
  std::string engineName_;
  std::vector<std::string> classNames_;
  float confThreshold_;
  int inputSize_;

  // TensorRT's IExecutionContext is NOT safe to call concurrently from
  // multiple threads. gRPC's default (sync) server dispatches each
  // incoming RPC on a thread from its own internal pool, so two Predict
  // calls arriving close together could otherwise race on the same
  // execution context and corrupt each other's inference. This mutex
  // serializes access to engine_->infer() so Predict is safe as written.
  std::mutex engineMutex_;
};
