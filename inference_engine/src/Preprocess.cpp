#include "Preprocess.hpp"

#include <algorithm>
#include <cmath>

std::vector<float> preprocessLetterbox(const cv::Mat &image, int targetSize,
                                       PreprocessMeta &metaOut) {
  const int origW = image.cols;
  const int origH = image.rows;
  // Scale to fit inside targetSize x targetSize, preserving aspect ratio.
  const float scale = std::min(static_cast<float>(targetSize) / origW,
                               static_cast<float>(targetSize) / origH);
  const int newW = static_cast<int>(std::round(origW * scale));
  const int newH = static_cast<int>(std::round(origH * scale));
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(newW, newH));
  // Center the resized image in a targetSize x targetSize canvas, padded
  // with gray
  const int padX = (targetSize - newW) / 2;
  const int padY = (targetSize - newH) / 2;
  cv::Mat padded(targetSize, targetSize, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(padX, padY, newW, newH)));
  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
  // HWC uint8 -> CHW float32, normalized to [0,1].
  std::vector<float> input(static_cast<size_t>(3) * targetSize * targetSize);
  const size_t planeSize = static_cast<size_t>(targetSize) * targetSize;
  for (int y = 0; y < targetSize; ++y) {
    for (int x = 0; x < targetSize; ++x) {
      const cv::Vec3b px = rgb.at<cv::Vec3b>(y, x);
      const size_t idx = static_cast<size_t>(y) * targetSize + x;
      input[0 * planeSize + idx] = px[0] / 255.0f; // R plane
      input[1 * planeSize + idx] = px[1] / 255.0f; // G plane
      input[2 * planeSize + idx] = px[2] / 255.0f; // B plane
    }
  }
  metaOut.scale = scale;
  metaOut.padX = padX;
  metaOut.padY = padY;
  metaOut.origWidth = origW;
  metaOut.origHeight = origH;
  return input;
}
