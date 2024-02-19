#!/bin/bash

# Navigate to the Protobuf directory
cd protobuf

# Compile Protobuf files
protoc --python_out=../src/telemetry_access_controller/generated \
       --grpc_python_out=../src/telemetry_access_controller/generated \
       telemetry.proto telemetry_service.proto

echo "Protobuf compilation completed."
