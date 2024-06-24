#!/bin/sh
set -e

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Create the pre-commit hook in the .git/hooks directory
HOOKS_DIR=".git/hooks"
PRE_COMMIT_HOOK="${HOOKS_DIR}/pre-commit"

mkdir -p $HOOKS_DIR

# Create a symlink to the validate.sh script in the .git/hooks directory
ln -sf ../../scripts/validate.sh $PRE_COMMIT_HOOK

# Make the pre-commit hook executable
chmod +x $PRE_COMMIT_HOOK

echo "Git pre-commit hook installed successfully."
