# Cramer: a Numerical Linear Algebra Library for C++

![build](https://github.com/cm-jones/cramer/actions/workflows/build.yml/badge.svg)
[![Test Coverage](https://codecov.io/gh/cm-jones/cramer/branch/main/graph/badge.svg?token=fc9ee083-78b6-4e43-bf23-bfa85832df85)](https://codecov.io/gh/cm-jones/cramer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Description

Cramer is an open-source, numerical linear algebra library for C++ that implements operations on vectors and matrices. It can also compute eigenvalues and eigenvectors and solve systems of linear equations. Here's an incomplete list of features:

- Vector operations:
  - norm
  - inner (dot) product
  - cross product
  - projections
  - reflections

- Matrix operations:
  - trace
  - determinant
  - transpose
  - inverse
  - LU, QR, and SVD decompositions
  - eigenvalues and eigenvectors

## Requirements

- C++17 or later
- CMake 3.10 or later

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

## Usage

Include the header in your C++ source files:

```cpp
#include <cramer.h>
```

Create vectors and matrices, and perform operations on them:

```cpp
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
```

## Documentation

Detailed documentation for cramer can be found at https://cm-jones.github.io/cramer/docs/.

## Contributing

If you find any issues or have suggestions for improvements, feel free to [open an issue](https://github.com/cm-jones/cramer/issues/new) or submit a [pull request](https://github.com/cm-jones/cramer/compare).

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
