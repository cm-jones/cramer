# Cramer: a Numerical Linear Algebra Library

![build](https://github.com/cm-jones/cramer/actions/workflows/ci.yml/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Cramer is an open-source, numerical linear algebra library for C++ that implements basic operations on vectors (e.g., norm, inner product, and outer product) and matrices (e.g., trace, determinant, transpose) as well as important algorithms in linear algebra (e.g., computing eigenvectors and eigenvalues, solving systems of linear equations).

## Features

Cramer supports the following computations:

- For any two vectors $x$ and $y$,
  - $\lVert x \rVert$, the norm,
  - $x \cdot y$, the inner product, and
  - $x \otimes y$, the outer product.

- For a square matrix $A_{n \times n}$,
  - $tr(A)$, the trace of A,
  - $det(A)$, the determinate of A,
  - $A^T$, the transpose of A,
  - $A^{-1}$, the inverse of A (if it exists),
  - $A^n$, the integer powers of A, and
  - the eigenvalues and eigenvectors of $A$, such that $Ax = \lambda x$.

- For any matrix $A_{m \times n}$,
  - the LU, QR, and SVD factorizations of $A$, and
  - solving a system of linear equations, such that $Ax = b$.

## Installation

### Requirements

- C++14 or later
- CMake 3.10 or later

### Build and Install

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

Include the necessary headers in your C++ source files:

```cpp
#include <cramer/vector.h>
#include <cramer/matrix.h>
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

Detailed documentation for cramer can be found under [docs/](docs/).

## Contributing

If you find any issues or have suggestions for improvements, feel free to [open an issue](https://github.com/cm-jones/cramer/issues/new) or submit a [pull request](https://github.com/cm-jones/cramer/compare).

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
