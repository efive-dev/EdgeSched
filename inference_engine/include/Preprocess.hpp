#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct PreprocessMeta {
  float scale = 1.0f;
  int padX = 0;
  int padY = 0;
  int origWidth = 0;
  int origHeight = 0;
};

std::vector<float> preprocessLetterbox(const cv::Mat &image, int targetSize,
                                       PreprocessMeta &metaOut);
