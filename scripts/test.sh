#!/bin/sh
set -e

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Navigate to the build directory
cd build

# Run tests
ctest --output-on-failure
