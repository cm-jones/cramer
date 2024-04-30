# `matrix.h`

## Matrix Class

The `Matrix` class represents a matrix of elements of type `T`. It provides various constructors, operators, and member functions for matrix manipulation and computation.

### Constructors

- `Matrix()`: Default constructor. Creates an empty matrix.
- `Matrix(size_t rows, size_t cols)`: Creates a matrix of a specified size.
- `Matrix(size_t rows, size_t cols, const T& value)`: Creates a matrix of a specified size and initializes all elements with a given value.
- `Matrix(const std::vector<std::vector<T>>& values)`: Creates a matrix from a 2D vector.

### Member Functions

- `size_t rows() const`: Returns the number of rows in the matrix.
- `size_t cols() const`: Returns the number of columns in the matrix.
- `T& operator()(size_t row, size_t col)`: Overloads the `()` operator to access elements of the matrix.
- `const T& operator()(size_t row, size_t col) const`: Overloads the `()` operator to access elements of the matrix (const version).
- `Matrix<T> operator*(const Matrix<T>& other) const`: Overloads the `*` operator to perform matrix multiplication.
- `Matrix<T> operator*(const T& scalar) const`: Overloads the `*` operator to perform scalar multiplication.
- `Matrix<T>& operator*=(const T& scalar)`: Overloads the `*=` operator to perform scalar multiplication and assignment.
- `Matrix<T> operator+(const Matrix<T>& other) const`: Overloads the `+` operator to perform matrix addition.
- `Matrix<T>& operator+=(const Matrix<T>& other)`: Overloads the `+=` operator to perform matrix addition and assignment.
- `Matrix<T> operator-(const Matrix<T>& other) const`: Overloads the `-` operator to perform matrix subtraction.
- `Matrix<T>& operator-=(const Matrix<T>& other)`: Overloads the `-=` operator to perform matrix subtraction and assignment.
- `std::pair<Matrix<T>, Matrix<T>> lu_decomp() const`: Performs LU decomposition on the matrix.
- `T det_lu() const`: Calculates the determinant of the matrix using LU decomposition.
- `std::vector<std::complex<T>> eigenvalues() const`: Calculates the eigenvalues of the matrix.
- `Matrix<T> eigenvectors() const`: Calculates the eigenvectors of the matrix.
- `Matrix<T> transpose() const`: Calculates the transpose of the matrix.
- `T trace() const`: Calculates the trace of the matrix.
- `T det() const`: Calculates the determinant of the matrix.
- `Matrix<T> inverse() const`: Calculates the inverse of the matrix.
- `Matrix<T> exp() const`: Calculates the matrix exponential of the matrix.
