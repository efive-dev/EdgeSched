#include "CocoClasses.hpp"
#include "inference.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

// Deterministically maps a class_id to a distinct, visually-spread color
static cv::Scalar colorForClass(int classId) {
  const float hueDeg = std::fmod(classId * 137.508f, 360.0f);
  // OpenCV's 8 bit HSV hue range is 0-179, not 0-359.
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hueDeg / 2.0f, 200, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(c[0], c[1], c[2]);
}

// Usage: ./test_client <host:port> <image_path>
// Sends the image to the server, prints detections, and saves an annotated
// copy (boxes + labels drawn, one color per class) next to the original.
int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <host:port> <image_path>"
              << std::endl;
    return 1;
  }
  const std::string target = argv[1];
  const std::string imagePath = argv[2];
  std::ifstream file(imagePath, std::ios::binary);
  if (!file) {
    std::cerr << "Cannot open image: " << imagePath << std::endl;
    return 1;
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  const std::string imageBytes = ss.str();
  auto channel =
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  auto stub = edgesched::Inference::NewStub(channel);
  // HealthCheck first
  {
    edgesched::HealthCheckRequest req;
    edgesched::HealthCheckResponse resp;
    grpc::ClientContext ctx;
    auto status = stub->HealthCheck(&ctx, req, &resp);
    if (!status.ok()) {
      std::cerr << "HealthCheck failed: " << status.error_message()
                << std::endl;
      return 1;
    }
    std::cout << "HealthCheck OK -- serving=" << resp.serving()
              << " engine_name=" << resp.engine_name() << std::endl;
  }
  // Predict
  edgesched::PredictRequest request;
  request.set_image_data(imageBytes);
  edgesched::PredictResponse response;
  grpc::ClientContext context;
  auto status = stub->Predict(&context, request, &response);
  if (!status.ok()) {
    std::cerr << "Predict failed: " << status.error_message() << std::endl;
    return 1;
  }
  std::cout << "\npreprocess_ms=" << response.preprocess_ms()
            << " inference_ms=" << response.inference_ms()
            << " postprocess_ms=" << response.postprocess_ms() << std::endl;
  std::cout << "\nDetections (" << response.detections_size()
            << "):" << std::endl;
  for (const auto &d : response.detections()) {
    std::cout << "  class_id=" << d.class_id() << " conf=" << d.confidence()
              << " box=[" << d.x1() << ", " << d.y1() << ", " << d.x2() << ", "
              << d.y2() << "]" << std::endl;
  }
  // Draw boxes on the original image and save an annotated copy
  std::vector<uchar> buf(imageBytes.begin(), imageBytes.end());
  cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Warning: could not decode image locally for drawing; "
                 "skipping annotated output."
              << std::endl;
    return 0;
  }
  const int thickness = 6;
  for (const auto &d : response.detections()) {
    const cv::Scalar color = colorForClass(d.class_id());
    cv::Point topLeft(static_cast<int>(d.x1()), static_cast<int>(d.y1()));
    cv::Point bottomRight(static_cast<int>(d.x2()), static_cast<int>(d.y2()));
    cv::rectangle(image, topLeft, bottomRight, color, thickness);
    std::string label =
        !d.class_name().empty() ? d.class_name() : cocoClassName(d.class_id());
    char confBuf[16];
    std::snprintf(confBuf, sizeof(confBuf), " %.2f", d.confidence());
    label += confBuf;
    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::Point labelOrigin(topLeft.x, std::max(topLeft.y - 4, textSize.height));
    cv::rectangle(
        image, cv::Point(labelOrigin.x, labelOrigin.y - textSize.height - 4),
        cv::Point(labelOrigin.x + textSize.width + 4, labelOrigin.y + 2), color,
        cv::FILLED);
    cv::putText(image, label, cv::Point(labelOrigin.x + 2, labelOrigin.y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
  std::string outPath = imagePath;
  auto dotPos = outPath.find_last_of('.');
  if (dotPos == std::string::npos) {
    outPath += "_annotated";
  } else {
    outPath = outPath.substr(0, dotPos) + "_annotated" + outPath.substr(dotPos);
  }
  if (cv::imwrite(outPath, image)) {
    std::cout << "\nAnnotated image saved to: " << outPath << std::endl;
  } else {
    std::cerr << "\nFailed to write annotated image to: " << outPath
              << std::endl;
  }
  return 0;
}
