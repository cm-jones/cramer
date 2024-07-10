// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <complex>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cramer {

/**
 * @brief Represents a matrix of elements of type T.
 *
 * @tparam T The type of elements stored in the matrix.
 */
template <typename T>
class Matrix {
   private:
    size_t rows;                            /**< The number of rows in the matrix. */
    size_t cols;                            /**< The number of columns in the matrix. */
    std::vector<std::vector<T>> data;       /**< The underlying container storing the matrix elements. */

   public:
    /**
     * @brief Default constructor. Creates an empty matrix.
     */
    Matrix();

    /**
     * @brief Constructor that creates a matrix of a specified size.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    Matrix(size_t rows, size_t cols);

    /**
     * @brief Constructor that creates a matrix of a specified size and initializes all elements with a given value.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param value The value to initialize all elements with.
     */
    Matrix(size_t rows, size_t cols, const T& value);

    /**
     * @brief Constructor that creates a matrix from a vector of vectors.
     *
     * @param values The vector of vectors representing the matrix elements.
     * @throws std::invalid_argument If the input vector is empty or the rows have different sizes.
     */
    Matrix(const std::vector<std::vector<T>>& values);

    /**
     * @brief Constructor that creates a matrix from an initializer list.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param values The initializer list representing the matrix elements.
     * @throws std::invalid_argument If the initializer list size does not match the specified dimensions.
     */
    Matrix(size_t rows, size_t cols, std::initializer_list<T> values);

    /**
     * @brief Gets the number of rows in the matrix.
     *
     * @return The number of rows in the matrix.
     */
    size_t get_rows() const;

    /**
     * @brief Gets the number of columns in the matrix.
     *
     * @return The number of columns in the matrix.
     */
    size_t get_cols() const;

    /**
     * @brief Creates an identity matrix of a specified size.
     *
     * @param size The size of the identity matrix.
     * @return The identity matrix.
     */
    static Matrix<T> identity(size_t size);

    /**
     * @brief Creates a matrix of zeros of a specified size.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @return The matrix of zeros.
     */
    static Matrix<T> zeros(size_t rows, size_t cols);

    /**
     * @brief Creates a matrix of ones of a specified size.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @return The matrix of ones.
     */
    static Matrix<T> ones(size_t rows, size_t cols);

    /**
     * @brief Creates a random matrix of a specified size.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @return The random matrix.
     * @throws std::runtime_error If the random matrix generation is not supported for the given type.
     */
    static Matrix<T> random(size_t rows, size_t cols);

    /**
     * @brief Overloads the () operator to access elements of the matrix.
     *
     * @param row The row index of the element to access.
     * @param col The column index of the element to access.
     * @return A reference to the element at the specified indices.
     */
    T& operator()(size_t row, size_t col);

    /**
     * @brief Overloads the () operator to access elements of the matrix (const version).
     *
     * @param row The row index of the element to access.
     * @param col The column index of the element to access.
     * @return A const reference to the element at the specified indices.
     */
    const T& operator()(size_t row, size_t col) const;

    /**
     * @brief Overloads the == operator to check for matrix equality.
     *
     * @param other The matrix to compare with.
     * @return true if the matrices are equal, false otherwise.
     */
    bool operator==(const Matrix<T>& other) const;

    /**
     * @brief Overloads the + operator to perform matrix addition.
     *
     * @param other The matrix to add to the current matrix.
     * @return A new matrix that is the result of adding the current matrix and the other matrix.
     * @throws std::invalid_argument If the matrices have different dimensions.
     */
    Matrix<T> operator+(const Matrix<T>& other) const;

    /**
     * @brief Overloads the += operator to perform matrix addition and assignment.
     *
     * @param other The matrix to add to the current matrix.
     * @return A reference to the updated current matrix.
     * @throws std::invalid_argument If the matrices have different dimensions.
     */
    Matrix<T>& operator+=(const Matrix<T>& other);

    /**
     * @brief Overloads the - operator to perform matrix subtraction.
     *
     * @param other The matrix to subtract from the current matrix.
     * @return A new matrix that is the result of subtracting the other matrix from the current matrix.
     * @throws std::invalid_argument If the matrices have different dimensions.
     */
    Matrix<T> operator-(const Matrix<T>& other) const;

    /**
     * @brief Overloads the -= operator to perform matrix subtraction and assignment.
     *
     * @param other The matrix to subtract from the current matrix.
     * @return A reference to the updated current matrix.
     * @throws std::invalid_argument If the matrices have different dimensions.
     */
    Matrix<T>& operator-=(const Matrix<T>& other);

    /**
     * @brief Overloads the * operator to perform scalar multiplication.
     *
     * @param scalar The scalar to multiply the matrix by.
     * @return A new matrix that is the result of multiplying the current matrix by the scalar.
     */
    Matrix<T> operator*(const T& scalar) const;

    /**
     * @brief Overloads the *= operator to perform scalar multiplication and assignment.
     *
     * @param scalar The scalar to multiply the matrix by.
     * @return A reference to the updated current matrix.
     */
    Matrix<T>& operator*=(const T& scalar);

    /**
     * @brief Overloads the * operator to perform matrix multiplication.
     *
     * @param other The matrix to multiply the current matrix by.
     * @return A new matrix that is the result of multiplying the current matrix by the other matrix.
     * @throws std::invalid_argument If the dimensions of the matrices do not match for multiplication.
     */
    Matrix<T> operator*(const Matrix<T>& other) const;

    /**
     * @brief Multiplies the matrix by a vector.
     *
     * @param vec The vector to multiply the matrix by.
     * @return The resulting vector after matrix-vector multiplication.
     * @throws std::invalid_argument If the dimensions of the matrix and vector do not match for multiplication.
     */
    Vector<T> multiply_vector(const Vector<T>& vec) const;

    /**
     * @brief Checks if the matrix is square.
     *
     * @return true if the matrix is square, false otherwise.
     */
    bool is_square() const;

    /**
     * @brief Checks if the matrix is diagonal.
     *
     * @return true if the matrix is diagonal, false otherwise.
     */
    bool is_diagonal() const;

    /**
     * @brief Checks if the matrix is symmetric.
     *
     * @return true if the matrix is symmetric, false otherwise.
     */
    bool is_symmetric() const;

    /**
     * @brief Checks if the matrix is invertible.
     *
     * @return true if the matrix is invertible, false otherwise.
     */
    bool is_invertible() const;

    /**
     * @brief Checks if the matrix is Hermitian (conjugate symmetric).
     *
     * @return true if the matrix is Hermitian, false otherwise.
     */
    bool is_hermitian() const;

    /**
     * @brief Checks if the matrix is orthogonal.
     *
     * @return true if the matrix is orthogonal, false otherwise.
     */
    bool is_orthogonal() const;

    /**
     * @brief Calculates the trace of the matrix.
     *
     * @return The trace of the matrix.
     */
    T trace() const;

    /**
     * @brief Calculates the determinant of the matrix.
     *
     * @return The determinant of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     */
    T det() const;

    /**
     * @brief Calculates the determinant of the matrix using LU decomposition.
     *
     * @return The determinant of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     */
    T det_via_lu() const;

    /**
     * @brief Calculates the rank of the matrix.
     *
     * @return The rank of the matrix.
     */
    size_t rank() const;

    /**
     * @brief Calculates the transpose of the matrix.
     *
     * @return The transpose of the matrix.
     */
    Matrix<T> transpose() const;

    /**
     * @brief Calculates the inverse of the matrix.
     *
     * @return The inverse of the matrix.
     * @throws std::invalid_argument If the matrix is not square or invertible.
     */
    Matrix<T> inverse() const;

    /**
     * @brief Calculates the adjoint (adjugate) of the matrix.
     *
     * @return The adjoint of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     */
    Matrix<T> adjoint() const;

    /**
     * @brief Calculates the conjugate of the matrix.
     *
     * @return The conjugate of the matrix.
     * @throws std::runtime_error If the conjugate operation is not supported for the given type.
     */
    Matrix<T> conjugate() const;

    /**
     * @brief Calculates the exponential of the matrix.
     *
     * @return The exponential of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     */
    Matrix<T> exp() const;

    /**
     * @brief Calculates the matrix raised to a specified integer power.
     *
     * @param n The power to raise the matrix to.
     * @return The matrix raised to the specified power.
     * @throws std::invalid_argument If the matrix is not square.
     */
    Matrix<T> pow(int n) const;

    /**
     * @brief Calculates the square root of the matrix.
     *
     * @return The square root of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     * @throws std::runtime_error If the square root iteration does not converge.
     */
    Matrix<T> sqrt() const;

    /**
     * @brief Calculates the natural logarithm of the matrix.
     *
     * @return The natural logarithm of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     */
    Matrix<T> log() const;

    /**
     * @brief Performs LU decomposition of the matrix.
     *
     * @return A pair containing the lower and upper triangular matrices (L and U).
     * @throws std::invalid_argument If the matrix is not square.
     */
    std::pair<Matrix<T>, Matrix<T>> lu() const;

    /**
     * @brief Performs QR decomposition of the matrix.
     *
     * @return A pair containing the orthogonal matrix Q and upper triangular matrix R.
     */
    std::pair<Matrix<T>, Matrix<T>> qr() const;

    /**
     * @brief Performs singular value decomposition (SVD) of the matrix.
     *
     * @return A tuple containing the left singular vectors (U), singular values (S), and right singular vectors (V).
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() const;

    /**
     * @brief Calculates the eigenvalues of the matrix.
     *
     * @return A vector containing the eigenvalues of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     */
    std::vector<std::complex<T>> eigenvalues() const;

    /**
     * @brief Calculates the eigenvectors of the matrix.
     *
     * @return A matrix containing the eigenvectors of the matrix.
     * @throws std::invalid_argument If the matrix is not square.
     */
    Matrix<T> eigenvectors() const;

    /**
     * @brief Solves a linear system of equations Ax = b using LU decomposition.
     *
     * @param b The right-hand side vector b.
     * @return The solution vector x.
     * @throws std::invalid_argument If the matrix is not square or the dimensions do not match.
     */
    Vector<T> solve(const Vector<T>& b) const;

   private:
    /**
     * @brief Calculates the maximum norm of the matrix.
     *
     * @return The maximum norm of the matrix.
     */
    T max_norm() const;
};

}  // namespace cramer
