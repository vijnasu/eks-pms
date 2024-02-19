#!/bin/bash

# Navigate to the Protobuf directory
cd protobuf

# Compile Protobuf files, including the path to the well-known types

python3 -m grpc_tools.protoc \
--proto_path=. \
--python_out=../src/telemetry_access_controller/generated \
--grpc_python_out=../src/telemetry_access_controller/generated tac.proto tac_service.proto

echo "Protobuf compilation completed."
