#!/bin/sh

# Update the PATH to include the directory where clang-tidy is installed
export PATH="/opt/homebrew/opt/llvm/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Navigate to the project root directory
cd "$(git rev-parse --show-toplevel)"

# Run clang-format and clang-tidy in parallel
./scripts/format.sh &
FORMAT_PID=$!

./scripts/lint.sh &
LINT_PID=$!

# Wait for both processes to complete
wait $FORMAT_PID
FORMAT_RESULT=$?

wait $LINT_PID
LINT_RESULT=$?

# Check if formatting and linting were successful before proceeding
if [ $FORMAT_RESULT -ne 0 ] || [ $LINT_RESULT -ne 0 ]; then
    echo "Pre-commit hook failed. Fix the errors before committing."
    exit 1
fi

# Build the project
./scripts/build.sh
BUILD_RESULT=$?

# Run tests
./scripts/test.sh
TEST_RESULT=$?

# Check if all checks were successful
if [ $BUILD_RESULT -ne 0 ] || [ $TEST_RESULT -ne 0 ]; then
    echo "Pre-commit hook failed. Fix the errors before committing."
    exit 1
fi

echo "Pre-commit hook passed. Proceeding with commit."
exit 0
