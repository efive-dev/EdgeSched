#include "Postprocess.hpp"
#include "Preprocess.hpp"

#include <gtest/gtest.h>

// postprocess no OpenCV/CUDA involved.
TEST(Postprocess, FiltersBelowConfidenceThreshold) {
  std::vector<float> raw = {
      10, 10, 20, 20, 0.9f, 1, // kept
      30, 30, 40, 40, 0.1f, 2, // dropped
  };
  PreprocessMeta meta{1.0f, 0, 0, 100, 100};
  auto results = postprocess(raw, meta, 0.25f);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].classId, 1);
}

TEST(Postprocess, UndoesLetterboxScaleAndPadding) {
  std::vector<float> raw = {110, 60, 210, 160, 0.9f, 0};
  PreprocessMeta meta{0.5f, 10, 20, 500, 500};
  auto results = postprocess(raw, meta, 0.25f);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FLOAT_EQ(results[0].x1, (110 - 10) / 0.5f);
  EXPECT_FLOAT_EQ(results[0].y1, (60 - 20) / 0.5f);
  EXPECT_FLOAT_EQ(results[0].x2, (210 - 10) / 0.5f);
  EXPECT_FLOAT_EQ(results[0].y2, (160 - 20) / 0.5f);
}

TEST(Postprocess, ClampsBoxesToOriginalImageBounds) {
  std::vector<float> raw = {0, 0, 640, 640, 0.9f, 0};
  PreprocessMeta meta{1.0f, 0, 0, 100, 50};
  auto results = postprocess(raw, meta, 0.25f);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FLOAT_EQ(results[0].x1, 0.0f);
  EXPECT_FLOAT_EQ(results[0].y1, 0.0f);
  EXPECT_FLOAT_EQ(results[0].x2, 100.0f);
  EXPECT_FLOAT_EQ(results[0].y2, 50.0f);
}

TEST(Postprocess, EmptyWhenNothingPassesThreshold) {
  std::vector<float> raw = {10, 10, 20, 20, 0.01f, 0};
  PreprocessMeta meta{1.0f, 0, 0, 100, 100};
  auto results = postprocess(raw, meta, 0.25f);
  EXPECT_TRUE(results.empty());
}

// preprocessLetterbox needs OpenCV, not CUDA/TensorRT.
TEST(Preprocess, OutputSizeMatchesTargetDimensions) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  PreprocessMeta meta;
  auto tensor = preprocessLetterbox(image, 640, meta);
  EXPECT_EQ(tensor.size(), static_cast<size_t>(3 * 640 * 640));
}

TEST(Preprocess, WideImageConstrainedByWidth) {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
  PreprocessMeta meta;
  preprocessLetterbox(image, 640, meta);
  EXPECT_FLOAT_EQ(meta.scale, 640.0f / 1280.0f);
  EXPECT_EQ(meta.padX, 0);
  EXPECT_EQ(meta.padY, (640 - 360) / 2);
}

TEST(Preprocess, TallImageConstrainedByHeight) {
  cv::Mat image(1280, 720, CV_8UC3, cv::Scalar(0, 0, 0));
  PreprocessMeta meta;
  preprocessLetterbox(image, 640, meta);
  EXPECT_FLOAT_EQ(meta.scale, 640.0f / 1280.0f);
  EXPECT_EQ(meta.padY, 0);
  EXPECT_EQ(meta.padX, (640 - 360) / 2);
}

TEST(Preprocess, SquareImageNeedsNoPadding) {
  cv::Mat image(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  PreprocessMeta meta;
  preprocessLetterbox(image, 640, meta);
  EXPECT_FLOAT_EQ(meta.scale, 1.0f);
  EXPECT_EQ(meta.padX, 0);
  EXPECT_EQ(meta.padY, 0);
}

TEST(Preprocess, NormalizesToZeroOneRangeInCorrectChannelOrder) {
  cv::Mat image(640, 640, CV_8UC3, cv::Scalar(0, 0, 255)); // pure red, BGR
  PreprocessMeta meta;
  auto tensor = preprocessLetterbox(image, 640, meta);
  const size_t planeSize = 640 * 640;
  const size_t centerIdx = 320 * 640 + 320;
  EXPECT_NEAR(tensor[0 * planeSize + centerIdx], 1.0f, 1e-3f); // R
  EXPECT_NEAR(tensor[1 * planeSize + centerIdx], 0.0f, 1e-3f); // G
  EXPECT_NEAR(tensor[2 * planeSize + centerIdx], 0.0f, 1e-3f); // B
}
