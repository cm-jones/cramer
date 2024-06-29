# Cramer: a Numerical Linear Algebra Library for C++

![build](https://github.com/cm-jones/cramer/actions/workflows/build.yml/badge.svg)
[![Test Coverage](https://codecov.io/gh/cm-jones/cramer/branch/main/graph/badge.svg?token=fc9ee083-78b6-4e43-bf23-bfa85832df85)](https://codecov.io/gh/cm-jones/cramer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Description

Cramer is an open-source, numerical linear algebra library for C++ that supports a large collection of computations with vectors and matrices. For a full list of features, see below.

If you're looking for a good (proof-based, non-computational) book on linear algebra, I highly recommend Sheldon Axler's [Linear Algebra Done Right](https://linear.axler.net/). This book will help you understand the essence of linear algebra from a mathematician's perspective.

## Features

Vector operations:

- norm (length)
- inner (dot) and cross product
- projections, reflections, and rotations

Matrix operations:

- rank, trace, and determinant
- transpose, adjoint, and inverse (if it exists)
- LU, QR, and SVD decompositions
- eigenvalues and eigenvectors

## Requirements

- C++17 compiler (e.g., GCC 7.0+, Clang 5.0+)
- CMake 3.12+
- Google Test 1.10.0+ for [unit testing](#unit-testing)
- Google Benchmark 1.6.1+ for [benchmarking](#benchmarking)

Optional:

- clang-format 10.0+ for code formatting
- clang-tidy 10.0+ for static analysis

## Usage

Include the header in your source files:

```cpp
#include <cramer>
```

Create vectors and matrices, and perform operations on them:

```cpp
#include <vector> // Not to be confused with cramer::Vector!

using namespace cramer;

Matrix<double> A(2, 2);
A(0, 0) = 1.0;
A(0, 1) = 2.0;
A(1, 0) = 3.0;
A(1, 1) = 4.0;

Vector<double> b(2);
b(0) = 5.0;
b(1) = 6.0;

Vector<double> x = solve(A, b);
std::vector<double> lambdas = A.eigenvalues();
```

## Build and Install

1. Clone the repository:
   ```
   git clone https://github.com/cm-jones/cramer.git
   ```

2. Create a build directory and navigate to it:
   ```
   cd cramer
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Build the library:
   ```
   make
   ```

5. (Optional) Install the library and headers:
   ```
   sudo make install
   ```

## Unit Testing

Cramer uses [Google Test](https://github.com/google/googletest) for unit testing. To run the unit tests, first [build](#build-and-install) the project, and then execute the following command inside the `build` directory:

```sh
ctest
```

## Benchmarking

Cramer uses [Google Benchmark](https://github.com/google/benchmark) for benchmarking. To run the benchmarks, follow these steps:

1. Build the benchmarks (if not already built):
   ```
   cmake --build build --target benchmark_all
   ```

2. Run the benchmarks:
   ```
   cd build
   make benchmark_all
   ./benchmark_all
   ```

## Documentation

Documentation is generated automatically via doxygen and GitHub Actions by the `.github/workflows/document.yml` workflow. It can be viewed at https://cm-jones.github.io/cramer/docs/html.

## Contributing

If you find any issues or have suggestions for improvements, feel free to [open an issue](https://github.com/cm-jones/cramer/issues/new) or submit a [pull request](https://github.com/cm-jones/cramer/compare).

To make contributions, the standard rules apply (see [Contributing to a Project](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) on GitHub):

1. Fork the repository.

2. Clone the fork.

3. Create a new branch with a descriptive name.

4. Commit and push your changes.

5. Create a pull request.

Before you start making changes, we recommend that you install a pre-commit hook via the `scripts/install_hooks.sh` script. This will ensure that each commit is properly built, tested, linted, and formatted.

To install the hook, you will first need to make all of the scripts executable by running the following command in the project's root directory:

```sh
chmod +x scripts/*.sh
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
