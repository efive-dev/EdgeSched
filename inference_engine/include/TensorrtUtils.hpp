#pragma once

#include <NvInfer.h>
#include <cstddef>

inline size_t dTypeSize(nvinfer1::DataType type) {
  switch (type) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kBOOL:
    return 1;
  default:
    return 4; // conservative fallback for any newer type
  }
}

// Returns element count of a tensor shape.
// Dims = [1, 3, 640, 640] -> 1 * 3 * 640 * 640
inline size_t volume(const nvinfer1::Dims &dimensions) {
  size_t volume = 1;
  for (int i = 0; i < dimensions.nbDims; i++) {
    volume *= static_cast<size_t>(dimensions.d[i]);
  }
  return volume;
}
