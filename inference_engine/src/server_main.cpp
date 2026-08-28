#include "InferenceServiceImpl.hpp"
#include "TensorrtEngine.hpp"

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Usage: ./inference_server <engine_path> <port> <engine_name> [confThreshold]
// One process per engine, per the project's architecture decision, run
// this once per model/quantization tier you want serving, each on its own
// port. e.g.:
//   ./inference_server yolo26n_int8.engine 50051 yolo26n_int8
//   ./inference_server yolo26m_fp16.engine 50052 yolo26m_fp16
int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <engine_path> <port> <engine_name> [confThreshold]"
              << std::endl;
    return 1;
  }
  const std::string enginePath = argv[1];
  const std::string port = argv[2];
  const std::string engineName = argv[3];
  const float confThreshold = argc > 4 ? std::stof(argv[4]) : 0.25f;
  std::vector<std::string> classNames;
  std::shared_ptr<TensorrtEngine> engine;
  try {
    engine = std::make_shared<TensorrtEngine>(enginePath);
  } catch (const std::exception &e) {
    std::cerr << "Failed to load engine '" << enginePath << "': " << e.what()
              << std::endl;
    return 1;
  }
  InferenceServiceImpl service(engine, engineName, classNames, confThreshold);
  const std::string serverAddr = "0.0.0.0:" + port;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(serverAddr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "Failed to start gRPC server on " << serverAddr << std::endl;
    return 1;
  }
  std::cout << "InferenceServiceImpl (" << engineName << ") listening on "
            << serverAddr << std::endl;
  server->Wait();
  return 0;
}
