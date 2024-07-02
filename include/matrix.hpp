// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <complex>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <vector>

namespace cramer {

template <typename T>
class Vector;

template <typename T>
class Matrix {
   private:
    size_t rows;
    size_t cols;
    std::vector<std::vector<T>> data;

    /**
     * @brief Calculates the maximum norm of the matrix.
     * @return The maximum norm value.
     */
    T max_norm() const;

   public:
    /**
     * @brief Default constructor. Creates an empty matrix.
     */
    Matrix();

    /**
     * @brief Constructor that creates a matrix with specified dimensions.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    Matrix(size_t rows, size_t cols);

    /**
     * @brief Constructor that creates a matrix with specified dimensions and
     * fills it with a value.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param value The value to fill the matrix with.
     */
    Matrix(size_t rows, size_t cols, const T& value);

    /**
     * @brief Constructor that creates a matrix from a 2D vector.
     * @param values The 2D vector to create the matrix from.
     */
    Matrix(const std::vector<std::vector<T>>& values);

    /**
     * @brief Constructor that creates a matrix from an initializer list.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param values The initializer list of values to populate the matrix.
     * @throws std::invalid_argument if the size of values doesn't match rows *
     * cols.
     */
    Matrix(size_t rows, size_t cols, std::initializer_list<T> values);

    /**
     * @brief Gets the number of rows in the matrix.
     * @return The number of rows.
     */
    size_t get_rows() const;

    /**
     * @brief Gets the number of columns in the matrix.
     * @return The number of columns.
     */
    size_t get_cols() const;

    /**
     * @brief Creates an identity matrix of specified size.
     * @param size The size of the square identity matrix.
     * @return The identity matrix.
     */
    static Matrix identity(size_t size);

    /**
     * @brief Creates a zero matrix of specified dimensions.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return The zero matrix.
     */
    static Matrix zeros(size_t rows, size_t cols);

    /**
     * @brief Creates a matrix of ones with specified dimensions.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return The matrix of ones.
     */
    static Matrix ones(size_t rows, size_t cols);

    /**
     * @brief Creates a matrix with random values between 0 and 1.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return The random matrix.
     */
    static Matrix random(size_t rows, size_t cols);

    /**
     * @brief Access operator for matrix elements.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the element at (row, col).
     */
    T& operator()(size_t row, size_t col);

    /**
     * @brief Const access operator for matrix elements.
     * @param row Row index.
     * @param col Column index.
     * @return Const reference to the element at (row, col).
     */
    const T& operator()(size_t row, size_t col) const;

    /**
     * @brief Equality comparison operator.
     * @param other The matrix to compare with.
     * @return True if matrices are equal, false otherwise.
     */
    bool operator==(const Matrix<T>& other) const;

    /**
     * @brief Addition operator.
     * @param other The matrix to add.
     * @return The sum of the two matrices.
     */
    Matrix<T> operator+(const Matrix<T>& other) const;

    /**
     * @brief Addition assignment operator.
     * @param other The matrix to add.
     * @return Reference to the modified matrix.
     */
    Matrix<T>& operator+=(const Matrix<T>& other);

    /**
     * @brief Subtraction operator.
     * @param other The matrix to subtract.
     * @return The difference of the two matrices.
     */
    Matrix<T> operator-(const Matrix<T>& other) const;

    /**
     * @brief Subtraction assignment operator.
     * @param other The matrix to subtract.
     * @return Reference to the modified matrix.
     */
    Matrix<T>& operator-=(const Matrix<T>& other);

    /**
     * @brief Scalar multiplication operator.
     * @param scalar The scalar to multiply by.
     * @return The product of the matrix and the scalar.
     */
    Matrix<T> operator*(const T& scalar) const;

    /**
     * @brief Scalar multiplication assignment operator.
     * @param scalar The scalar to multiply by.
     * @return Reference to the modified matrix.
     */
    Matrix<T>& operator*=(const T& scalar);

    /**
     * @brief Matrix multiplication operator.
     * @param other The matrix to multiply with.
     * @return The product of the two matrices.
     */
    Matrix<T> operator*(const Matrix<T>& other) const;

    /**
     * @brief Multiplies the matrix with a vector.
     * @param vec The vector to multiply with.
     * @return The resulting vector.
     */
    Vector<T> multiply_vector(const Vector<T>& vec) const;

    /**
     * @brief Checks if the matrix is square.
     * @return True if the matrix is square, false otherwise.
     */
    bool is_square() const;

    /**
     * @brief Checks if the matrix is diagonal.
     * @return True if the matrix is diagonal, false otherwise.
     */
    bool is_diagonal() const;

    /**
     * @brief Checks if the matrix is symmetric.
     * @return True if the matrix is symmetric, false otherwise.
     */
    bool is_symmetric() const;

    /**
     * @brief Checks if the matrix is invertible.
     * @return True if the matrix is invertible, false otherwise.
     */
    bool is_invertible() const;

    /**
     * @brief Checks if the matrix is Hermitian.
     * @return True if the matrix is Hermitian, false otherwise.
     */
    bool is_hermitian() const;

    /**
     * @brief Checks if the matrix is orthogonal.
     * @return True if the matrix is orthogonal, false otherwise.
     */
    bool is_orthogonal() const;

    /**
     * @brief Calculates the trace of the matrix.
     * @return The trace of the matrix.
     */
    T trace() const;

    /**
     * @brief Calculates the determinant of the matrix.
     * @return The determinant of the matrix.
     */
    T det() const;

    /**
     * @brief Calculates the determinant of the matrix using LU decomposition.
     * @return The determinant of the matrix.
     */
    T det_via_lu() const;

    /**
     * @brief Calculates the transpose of the matrix.
     * @return The transposed matrix.
     */
    Matrix<T> transpose() const;

    /**
     * @brief Calculates the inverse of the matrix.
     * @return The inverted matrix.
     */
    Matrix<T> inverse() const;

    /**
     * @brief Calculates the adjoint of the matrix.
     * @return The adjoint matrix.
     */
    Matrix<T> adjoint() const;

    /**
     * @brief Calculates the conjugate of the matrix.
     * @return The conjugate matrix.
     */
    Matrix<T> conjugate() const;

    /**
     * @brief Calculates the matrix exponential.
     * @return The exponential of the matrix.
     */
    Matrix<T> exp() const;

    /**
     * @brief Calculates the matrix power.
     * @param n The power to raise the matrix to.
     * @return The matrix raised to the power n.
     */
    Matrix<T> pow(int n) const;

    /**
     * @brief Calculates the square root of the matrix.
     * @return The square root of the matrix.
     */
    Matrix<T> sqrt() const;

    /**
     * @brief Performs LU decomposition of the matrix.
     * @return A pair of matrices (L, U) where L is lower triangular and U is
     * upper triangular.
     */
    std::pair<Matrix<T>, Matrix<T>> lu() const;

    /**
     * @brief Performs QR decomposition of the matrix.
     * @return A pair of matrices (Q, R) where Q is orthogonal and R is upper
     * triangular.
     */
    std::pair<Matrix<T>, Matrix<T>> qr() const;

    /**
     * @brief Performs Singular Value Decomposition (SVD) of the matrix.
     * @return A tuple of matrices (U, S, V) where U and V are orthogonal and S
     * is diagonal.
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() const;

    /**
     * @brief Calculates the eigenvalues of the matrix.
     * @return A vector of complex eigenvalues.
     */
    std::vector<std::complex<T>> eigenvalues() const;

    /**
     * @brief Calculates the eigenvectors of the matrix.
     * @return A matrix where each column is an eigenvector.
     */
    Matrix<T> eigenvectors() const;

    /**
     * @brief Calculates the rank of the matrix.
     * @return The rank of the matrix.
     */
    size_t rank() const;

    /**
     * @brief Solves the linear system Ax = b.
     * @param b The right-hand side vector.
     * @return The solution vector x.
     */
    Vector<T> solve(const Vector<T>& b) const;
};

}  // namespace cramer
