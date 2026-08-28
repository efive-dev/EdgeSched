#pragma once

#include "Preprocess.hpp"
#include <vector>

struct DetectionResult {
  float x1, y1, x2, y2;
  float confidence;
  int classId;
};

std::vector<DetectionResult> postprocess(const std::vector<float> &rawOutput,
                                         const PreprocessMeta &meta,
                                         float confThreshold);
