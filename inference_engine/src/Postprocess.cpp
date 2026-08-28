#include "Postprocess.hpp"

#include <algorithm>

std::vector<DetectionResult> postprocess(const std::vector<float> &rawOutput,
                                         const PreprocessMeta &meta,
                                         float confThreshold) {
  std::vector<DetectionResult> results;
  const int numCandidates = static_cast<int>(rawOutput.size() / 6);
  results.reserve(static_cast<size_t>(numCandidates));
  for (int i = 0; i < numCandidates; ++i) {
    const float *row = &rawOutput[static_cast<size_t>(i) * 6];
    const float conf = row[4];
    if (conf < confThreshold)
      continue;
    float x1 = row[0];
    float y1 = row[1];
    float x2 = row[2];
    float y2 = row[3];
    const int classId = static_cast<int>(row[5]);
    // Undo the letterbox transform: subtract padding, then divide by
    // the resize scale, to get back to original image pixel space.
    x1 = (x1 - meta.padX) / meta.scale;
    y1 = (y1 - meta.padY) / meta.scale;
    x2 = (x2 - meta.padX) / meta.scale;
    y2 = (y2 - meta.padY) / meta.scale;
    // Clamp in case the box extended into the padding region.
    x1 = std::clamp(x1, 0.0f, static_cast<float>(meta.origWidth));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(meta.origHeight));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(meta.origWidth));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(meta.origHeight));
    results.push_back(DetectionResult{x1, y1, x2, y2, conf, classId});
  }

  return results;
}
