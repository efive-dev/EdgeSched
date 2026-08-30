#include "Preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

std::vector<float> preprocessLetterbox(const cv::Mat &image, int targetSize,
                                       PreprocessMeta &metaOut) {
  const int origW = image.cols;
  const int origH = image.rows;
  const float scale = std::min(static_cast<float>(targetSize) / origW,
                               static_cast<float>(targetSize) / origH);
  const int newW = static_cast<int>(std::round(origW * scale));
  const int newH = static_cast<int>(std::round(origH * scale));
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(newW, newH));
  const int padX = (targetSize - newW) / 2;
  const int padY = (targetSize - newH) / 2;
  cv::Mat padded(targetSize, targetSize, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(padX, padY, newW, newH)));
  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
  // HWC uint8 -> CHW float32 [0,1], vectorized
  cv::Mat rgbFloat;
  rgb.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);
  std::vector<cv::Mat> channels(3);
  cv::split(rgbFloat, channels); // channels[0]=R, [1]=G, [2]=B
  std::vector<float> input(static_cast<size_t>(3) * targetSize * targetSize);
  const size_t planeSize = static_cast<size_t>(targetSize) * targetSize;
  for (int c = 0; c < 3; ++c) {
    CV_Assert(channels[c].isContinuous());
    std::memcpy(input.data() + static_cast<size_t>(c) * planeSize,
                channels[c].ptr<float>(0), planeSize * sizeof(float));
  }
  metaOut.scale = scale;
  metaOut.padX = padX;
  metaOut.padY = padY;
  metaOut.origWidth = origW;
  metaOut.origHeight = origH;
  return input;
}
