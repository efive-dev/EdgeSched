#include "TensorrtEngine.hpp"

#include <algorithm>
#include <iostream>
#include <random>

// Loads the engine, runs one inference pass on random input data, and
// prints the output tensor shapes/values.
// Proves that the loading of engine to execution works, rubbish results.
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <engine_path>" << std::endl;
    return 1;
  }
  try {
    TensorrtEngine engine(argv[1]);
    // Find the input tensor's expected element count so we can build
    // dummy data of the right size.
    size_t inputFloats = 0;
    for (const auto &t : engine.getTensorInfos()) {
      if (t.isInput) {
        inputFloats = t.sizeBytes / sizeof(float);
      }
    }
    if (inputFloats == 0) {
      std::cerr << "Could not determine input tensor size." << std::endl;
      return 1;
    }
    std::vector<float> dummyInput(inputFloats);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : dummyInput)
      v = dist(rng);
    std::cout << "\nRunning inference on random input (" << inputFloats
              << " floats)" << std::endl;
    auto outputs = engine.infer(dummyInput);
    std::cout << "\nInference succeeded. Got " << outputs.size()
              << " output tensor(s):" << std::endl;
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::cout << "  Output " << i << ": " << outputs[i].size()
                << " floats. First 5 values: ";
      for (size_t j = 0; j < std::min<size_t>(5, outputs[i].size()); ++j) {
        std::cout << outputs[i][j] << " ";
      }
      std::cout << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
