# Proto
The contract between the C++ engines and the Go control pane and scheduler is specified using *grpc* and *protobuf*. The main 
reasons that combo was chosen instead of a more standard rest api are:
- strong typing across languages;
- codegen for both languages, which means changes to the configuration file propagate everywhere;
- carrying images in JSON is complicated;
- Go + grpc is a fairly established pattern in the ML / infra space.
---

## Message design
| Message                   | Purpose                                            | Design                                                                                                                                                               |
| ------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`PredictRequest`**      | Sends an image to the inference service            | Carries the raw encoded image as `bytes` (e.g. JPEG/PNG). Image decoding remains the responsibility of the inference service rather than the client or scheduler.    |
| **`Detection`**           | Represents a single detected object                | Contains the detected object's bounding-box coordinates in **original image pixel space** (`x1`, `y1`, `x2`, `y2`), along with `class_id` and optional `class_name`. |
| **`PredictResponse`**     | Returns inference results                          | Contains the detections produced by the inference service and separate timing measurements for **preprocessing**, **inference**, and **postprocessing**.             |
| **`HealthCheckRequest`**  | Requests the health status of an inference service | Lightweight request used by the scheduler to check whether an engine is available and serving.                                                                       |
| **`HealthCheckResponse`** | Reports inference service health                   | Includes the serving status and `engine_name`, allowing the scheduler to identify which engine is healthy without relying on port to engine mappings.                |
---

## Services
For now only two rpcs are considered:
- Predict, which is the actual workload;
- HealthCheck, which is the way the go control pane monitors performance.

In the future additional requests could be implemented such as something to stream video or to warmup the models.
