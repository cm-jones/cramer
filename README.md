# LinCPP (Numerical Linear Algebra for C++)

LinCPP is an open-source, numerical linear algebra library for C++ that implements common operations on vectors and matrices, as well as important algorithms such as: solving a system of linear equations; LU, QR, and SVD decomposition; and computing the eigenvalues and eigenvectors of a square matrix.

## Features

- Computing the trace, determinant, transpose, inverse, and exponential of a matrix
- LU, QR, and SVD decomposition
- Solving a system of linear equations (Ax = b)
- Computing the eigenvalues and eigenvectors of a matrix A, satisfying Ax = lambda x

## Installation

### Requirements

- C++14 or later
- CMake 3.10 or later

### Build and Install

1. Clone the repository:
   ```
   git clone https://github.com/cm-jones/lincpp.git
   ```

2. Create a build directory and navigate to it:
   ```
   cd lincpp
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
#include <numcpp/vector.h>
#include <numcpp/matrix.h>
```

Create vectors and matrices, and perform operations on them:

```cpp
using namespace lincpp;

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

Detailed documentation for LinCPP can be found under [docs/](docs/).

## Contributing

If you find any issues or have suggestions for improvements, feel free to [open an issue](https://github.com/cm-jones/lincpp/issues/new) or submit a [pull request](https://github.com/cm-jones/lincpp/compare).

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
