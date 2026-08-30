#include "InferenceServiceImpl.hpp"

#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>

InferenceServiceImpl::InferenceServiceImpl(
    std::shared_ptr<TensorrtEngine> engine, std::string engineName,
    std::vector<std::string> classNames, float confThreshold, int inputSize)
    : engine_(std::move(engine)), engineName_(std::move(engineName)),
      classNames_(std::move(classNames)), confThreshold_(confThreshold),
      inputSize_(inputSize) {}

grpc::Status
InferenceServiceImpl::Predict(grpc::ServerContext * /*context*/,
                              const edgesched::PredictRequest *request,
                              edgesched::PredictResponse *response) {
  using clock = std::chrono::steady_clock;
  // Decode the incoming image bytes
  // Decoding at full resolution then immediately downscaling to
  // inputSize_ x inputSize
  const std::string &raw = request->image_data();
  std::vector<uchar> buf(raw.begin(), raw.end());
  cv::Mat image = cv::imdecode(buf, cv::IMREAD_REDUCED_COLOR_2);
  float decodeScaleFactor = 2.0f;
  if (!image.empty() && std::min(image.cols, image.rows) < inputSize_) {
    image = cv::imdecode(buf, cv::IMREAD_COLOR);
    decodeScaleFactor = 1.0f;
  }
  if (image.empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "failed to decode image data");
  }
  // Preprocess
  const auto t0 = clock::now();
  PreprocessMeta meta;
  std::vector<float> inputTensor = preprocessLetterbox(image, inputSize_, meta);
  const auto t1 = clock::now();
  // Inference
  std::vector<std::vector<float>> outputs;
  {
    std::lock_guard<std::mutex> lock(engineMutex_);
    outputs = engine_->infer(inputTensor);
  }
  const auto t2 = clock::now();
  if (outputs.empty()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "engine produced no output tensors");
  }
  // Postprocess
  auto detections = postprocess(outputs[0], meta, confThreshold_);
  const auto t3 = clock::now();
  for (const auto &d : detections) {
    auto *det = response->add_detections();
    // Scale back up to true original image pixel space
    det->set_x1(d.x1 * decodeScaleFactor);
    det->set_y1(d.y1 * decodeScaleFactor);
    det->set_x2(d.x2 * decodeScaleFactor);
    det->set_y2(d.y2 * decodeScaleFactor);
    det->set_confidence(d.confidence);
    det->set_class_id(d.classId);
    if (d.classId >= 0 && static_cast<size_t>(d.classId) < classNames_.size()) {
      det->set_class_name(classNames_[static_cast<size_t>(d.classId)]);
    }
  }
  response->set_preprocess_ms(
      std::chrono::duration<float, std::milli>(t1 - t0).count());
  response->set_inference_ms(
      std::chrono::duration<float, std::milli>(t2 - t1).count());
  response->set_postprocess_ms(
      std::chrono::duration<float, std::milli>(t3 - t2).count());
  return grpc::Status::OK;
}

grpc::Status InferenceServiceImpl::HealthCheck(
    grpc::ServerContext * /*context*/,
    const edgesched::HealthCheckRequest * /*request*/,
    edgesched::HealthCheckResponse *response) {
  response->set_serving(true);
  response->set_engine_name(engineName_);
  return grpc::Status::OK;
}
