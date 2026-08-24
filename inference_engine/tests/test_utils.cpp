#include "./../include/TensorrtUtils.hpp"
#include <gtest/gtest.h>

static nvinfer1::Dims makeDims(std::initializer_list<int64_t> vals) {
  nvinfer1::Dims dimensions{};
  dimensions.nbDims = static_cast<int>(vals.size());
  int i = 0;
  for (auto v : vals)
    dimensions.d[i++] = v;
  return dimensions;
}

// Tests for dTypeSize
TEST(DtypeSize, Float) { EXPECT_EQ(dTypeSize(nvinfer1::DataType::kFLOAT), 4u); }

TEST(DtypeSize, Half) { EXPECT_EQ(dTypeSize(nvinfer1::DataType::kHALF), 2u); }

TEST(DtypeSize, Int8) { EXPECT_EQ(dTypeSize(nvinfer1::DataType::kINT8), 1u); }

TEST(DtypeSize, Int32) { EXPECT_EQ(dTypeSize(nvinfer1::DataType::kINT32), 4u); }

TEST(DtypeSize, Bool) { EXPECT_EQ(dTypeSize(nvinfer1::DataType::kBOOL), 1u); }

// Volume
TEST(Volume, YoloInputShape) {
  auto dims = makeDims({1, 3, 640, 640});
  EXPECT_EQ(volume(dims), static_cast<size_t>(1 * 3 * 640 * 640));
}

TEST(Volume, YoloOutputShape) {
  auto dims = makeDims({1, 300, 6});
  EXPECT_EQ(volume(dims), static_cast<size_t>(1 * 300 * 6));
}

TEST(Volume, SingleDimension) {
  auto dims = makeDims({10});
  EXPECT_EQ(volume(dims), 10u);
}

TEST(Volume, BatchGreaterThanOne) {
  // Sanity check that batch size actually multiplies in
  auto dims = makeDims({4, 3, 224, 224});
  EXPECT_EQ(volume(dims), static_cast<size_t>(4 * 3 * 224 * 224));
}
