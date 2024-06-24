/*
 * This file is part of cramer.
 *
 * cramer is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * cramer is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * cramer. If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <complex>
#include <vector>

#include "vector.h"

namespace cramer {

/**
 * @brief Represents a matrix of elements of type T.
 *
 * @tparam T The type of elements stored in the matrix.
 */
template <typename T>
class Matrix {
   private:
    std::vector<std::vector<T>>
        data;    /**< The underlying container storing the matrix elements. */
    size_t rows; /**< The number of rows in the matrix. */
    size_t cols; /**< The number of columns in the matrix. */

    /**
     * @brief Calculates the maximum norm of the matrix.
     *
     * The maximum norm of a matrix is the maximum absolute row sum of the
     * matrix. It is defined as the maximum of the sums of the absolute values
     * of the elements in each row.
     *
     * @return The maximum norm of the matrix.
     */
    T max_norm() const;

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
     * @brief Constructor that creates a matrix of a specified size and
     * initializes all elements with a given value.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param value The value to initialize all elements with.
     */
    Matrix(size_t rows, size_t cols, const T& value);

    /**
     * @brief Constructor that creates a matrix from a 2D vector.
     *
     * @param values The 2D vector to create the matrix from.
     */
    Matrix(const std::vector<std::vector<T>>& values);

    /**
     * @brief Returns the number of rows in the matrix.
     *
     * @return The number of rows in the matrix.
     */
    size_t get_rows() const;

    /**
     * @brief Returns the number of columns in the matrix.
     *
     * @return The number of columns in the matrix.
     */
    size_t get_cols() const;

    /**
     * @brief Generates an identity matrix of a specified size.
     *
     * An identity matrix is a square matrix with ones on the main diagonal and
     * zeros elsewhere. It has the property that when multiplied with another
     * matrix, it leaves the other matrix unchanged.
     *
     * @param size The size of the identity matrix.
     * @return An identity matrix of the specified size.
     */
    static Matrix<T> identity(size_t size);

    /**
     * @brief Creates a matrix of the specified size with all elements
     * initialized to zero.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @return A matrix of the specified size with all elements set to zero.
     */
    static Matrix<T> zeros(size_t rows, size_t cols);

    /**
     * @brief Creates a matrix of the specified size with all elements
     * initialized to one.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @return A matrix of the specified size with all elements set to one.
     */
    static Matrix<T> ones(size_t rows, size_t cols);

    /**
     * @brief Creates a matrix of the specified size with elements initialized
     * to random values.
     *
     * The random values are generated using a uniform distribution between 0
     * and 1.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @return A matrix of the specified size with elements initialized to
     * random values.
     */
    static Matrix<T> random(size_t rows, size_t cols);

    /**
     * @brief Overloads the () operator to access elements of the matrix.
     *
     * @param row The row index of the element to access.
     * @param col The column index of the element to access.
     * @return A reference to the element at the specified row and column.
     */
    T& operator()(size_t row, size_t col);

    /**
     * @brief Overloads the () operator to access elements of the matrix (const
     * version).
     *
     * @param row The row index of the element to access.
     * @param col The column index of the element to access.
     * @return A const reference to the element at the specified row and column.
     */
    const T& operator()(size_t row, size_t col) const;

    /**
     * @brief Overloads the == operator to compare two matrices for equality.
     *
     * Two matrices are considered equal if they have the same dimensions and
     * all corresponding elements are equal.
     *
     * @param other The matrix to compare with.
     * @return True if the matrices are equal, false otherwise.
     */
    bool operator==(const Matrix<T>& other) const;

    /**
     * @brief Overloads the + operator to perform matrix addition.
     *
     * Matrix addition is an elementwise operation where corresponding elements
     * of two matrices are added together. The matrices must have the same
     * dimensions for addition to be valid.
     *
     * @param other The matrix to add to the current matrix.
     * @return The result of matrix addition.
     */
    Matrix<T> operator+(const Matrix<T>& other) const;

    /**
     * @brief Overloads the += operator to perform matrix addition and
     * assignment.
     *
     * @param other The matrix to add to the current matrix.
     * @return A reference to the updated matrix.
     */
    Matrix<T>& operator+=(const Matrix<T>& other);

    /**
     * @brief Overloads the - operator to perform matrix subtraction.
     *
     * Matrix subtraction is an elementwise operation where corresponding
     * elements of two matrices are subtracted. The matrices must have the same
     * dimensions for subtraction to be valid.
     *
     * @param other The matrix to subtract from the current matrix.
     * @return The result of matrix subtraction.
     */
    Matrix<T> operator-(const Matrix<T>& other) const;

    /**
     * @brief Overloads the -= operator to perform matrix subtraction and
     * assignment.
     *
     * @param other The matrix to subtract from the current matrix.
     * @return A reference to the updated matrix.
     */
    Matrix<T>& operator-=(const Matrix<T>& other);

    /**
     * @brief Overloads the * operator to perform scalar multiplication.
     *
     * Scalar multiplication is an operation where each element of a matrix is
     * multiplied by a scalar value.
     *
     * @param scalar The scalar value to multiply the matrix by.
     * @return The result of scalar multiplication.
     */
    Matrix<T> operator*(const T& scalar) const;

    /**
     * @brief Overloads the *= operator to perform scalar multiplication and
     * assignment.
     *
     * @param scalar The scalar value to multiply the matrix by.
     * @return A reference to the updated matrix.
     */
    Matrix<T>& operator*=(const T& scalar);

    /**
     * @brief Overloads the * operator to perform matrix multiplication.
     *
     * Matrix multiplication is a binary operation that produces a matrix from
     * two matrices. For matrix multiplication to be valid, the number of
     * columns in the first matrix must be equal to the number of rows in the
     * second matrix. The resulting matrix has the same number of rows as the
     * first matrix and the same number of columns as the second matrix.
     *
     * @param other The matrix to multiply with.
     * @return The result of matrix multiplication.
     */
    Matrix<T> operator*(const Matrix<T>& other) const;

    /**
     * @brief Determines if the matrix is square.
     *
     * @return true if the matrix is square
     * @return false if the matrix is not square
     */
    bool is_square() const;

    /**
     * @brief Determines if the matrix is diagonal.
     *
     * A diagonal matrix is a square matrix in which the entries outside the
     * main diagonal are all zero. The main diagonal is the diagonal from the
     * upper left to the lower right corner.
     *
     * @return True if the matrix is diagonal, false otherwise.
     */
    bool is_diagonal() const;

    /**
     * @brief Determines if the matrix is symmetric.
     *
     * A symmetric matrix is a square matrix that is equal to its transpose.
     * In other words, the element at row i and column j is equal to the
     * element at row j and column i, for all i and j.
     *
     * @return True if the matrix is symmetric, false otherwise.
     */
    bool is_symmetric() const;

    /**
     * @brief Determines if the matrix is invertible.
     *
     * @return true if the matrix is invertible
     * @return false if the matrix is not invertible
     */
    bool is_invertible() const;

    /**
     * @brief Determines if the matrix is Hermitian.
     *
     * A Hermitian matrix is a square matrix that is equal to its own conjugate
     * transpose. In other words, the element at row i and column j is equal to
     * the complex conjugate of the element at row j and column i, for all i
     * and j. For real matrices, a Hermitian matrix is the same as a symmetric
     * matrix.
     *
     * @return True if the matrix is Hermitian, false otherwise.
     */
    bool is_hermitian() const;

    /**
     * @brief Determines if a matrix is orthogonal.
     *
     * An orthogonal matrix is a square matrix whose columns and rows are
     * orthonormal vectors. In other words, the transpose of an orthogonal
     * matrix is equal to its inverse. Orthogonal matrices have the property
     * that they preserve the dot product of vectors and the length of vectors.
     *
     * @return True if the matrix is orthogonal, false otherwise.
     */
    bool is_orthogonal() const;

    /**
     * @brief Calculates the trace of the matrix.
     *
     * The trace of a matrix is the sum of the elements on the main diagonal
     * (from the upper left to the lower right). The matrix must be square for
     * the trace to be defined.
     *
     * @return The trace of the matrix.
     */
    T trace() const;

    /**
     * @brief Calculates the determinant of the matrix.
     *
     * The determinant of a matrix is a scalar value that can be computed from
     * the elements of a square matrix. It provides information about the
     * matrix, such as whether the matrix is invertible and the volume of the
     * linear transformation described by the matrix. The determinant is
     * calculated recursively using the Laplace expansion along the first row
     * of the matrix.
     *
     * @return The determinant of the matrix.
     */
    T det() const;

    /**
     * @brief Calculates the determinant of the matrix using LU decomposition.
     *
     * LU decomposition factorizes a matrix into a lower triangular matrix (L)
     * and an upper triangular matrix (U), such that A = LU. The determinant of
     * a triangular matrix is the product of its diagonal elements. This method
     * computes the determinant by calculating the LU decomposition and then
     * multiplying the diagonal elements of the upper triangular matrix.
     *
     * @return The determinant of the matrix.
     */
    T det_via_lu() const;

    /**
     * @brief Calculates the transpose of the matrix.
     *
     * The transpose of a matrix is an operator which flips a matrix over its
     * diagonal, that is, it switches the row and column indices of the matrix.
     * For example, the element at row i and column j in the original matrix
     * becomes the element at row j and column i in the transposed matrix.
     *
     * @return The transpose of the matrix.
     */
    Matrix<T> transpose() const;

    /**
     * @brief Calculates the inverse of the matrix.
     *
     * The inverse of a square matrix A is a matrix A^(-1) such that the
     * product of A and A^(-1) is the identity matrix. Not all matrices have an
     * inverse. A matrix is invertible (non-singular) if and only if its
     * determinant is non-zero. This method calculates the inverse using the
     * adjugate matrix and the determinant.
     *
     * @return The inverse of the matrix.
     */
    Matrix<T> inverse() const;

    /**
     * @brief Calculates the adjoint of the matrix.
     *
     * The adjoint of a matrix is the transpose of its cofactor matrix.
     * The cofactor matrix is obtained by replacing each element of the
     * original matrix by its cofactor. The cofactor of an element is the
     * product of its minor and the sign determined by the sum of its row and
     * column indices.
     *
     * @return The adjoint of the matrix.
     */
    Matrix<T> adjoint() const;

    /**
     * @brief Calculates the complex conjugate of the matrix.
     *
     * The complex conjugate of a matrix is obtained by replacing each element
     * with its complex conjugate. For real matrices, the complex conjugate is
     * the same as the original matrix.
     *
     * @return The complex conjugate of the matrix.
     */
    Matrix<T> conjugate() const;

    /**
     * @brief Calculates the matrix exponential of the matrix.
     *
     * The matrix exponential is a matrix function that generalizes the notion
     * of the exponential function to matrices. It is defined by the power
     * series exp(A) = I + A + A^2/2! + A^3/3! + ..., where I is the identity
     * matrix. The matrix exponential has many applications in solving systems
     * of linear differential equations.
     *
     * @return The matrix exponential of the matrix.
     */
    Matrix<T> exp() const;

    /**
     * @brief Calculates the integer power of a matrix.
     *
     * This function calculates the integer power of a matrix using the
     * exponentiation by squaring algorithm. It supports both positive and
     * negative integer exponents.
     *
     * @param n The integer exponent.
     * @return The matrix raised to the power of n.
     * @throws std::invalid_argument If the matrix is not square.
     */
    Matrix<T> pow(int n) const;

    /**
     * @brief Calculates the logarithm of a matrix.
     *
     * This function calculates the logarithm of a matrix using the inverse
     * scaling and squaring method. It uses Padé approximation to compute the
     * logarithm of the scaled matrix.
     *
     * @return The logarithm of the matrix.
     * @throws std::invalid_argument If the matrix is not square or is singular.
     */
    Matrix<T> log() const;

    /**
     * @brief Performs LU decomposition on the matrix.
     *
     * LU decomposition factorizes a matrix into a lower triangular matrix (L)
     * and an upper triangular matrix (U), such that A = LU. This method uses
     * Gaussian elimination with partial pivoting to compute the LU
     * decomposition.
     *
     * @return A pair of matrices (L, U) where L is a lower triangular matrix
     * and U is an upper triangular matrix.
     */
    std::pair<Matrix<T>, Matrix<T>> lu() const;

    /**
     * @brief Performs QR decomposition on the matrix.
     *
     * QR decomposition factorizes a matrix into an orthogonal matrix (Q) and
     * an upper triangular matrix (R), such that A = QR. This method uses the
     * Gram-Schmidt process to compute the QR decomposition.
     *
     * @return A*
     * @return A pair of matrices (Q, R) where Q is an orthogonal matrix and R
     * is an upper triangular matrix.
     */
    std::pair<Matrix<T>, Matrix<T>> qr() const;

    /**
     * @brief Performs SVD decomposition on the matrix.
     *
     * Singular Value Decomposition (SVD) factorizes a matrix into the product
     * of three matrices: A = U * Sigma * V^T, where U and V are orthogonal
     * matrices, and Sigma is a diagonal matrix containing the singular values.
     * SVD has many applications in signal processing, data compression, and
     * dimensionality reduction.
     *
     * @return A tuple of matrices (U, Sigma, V) where U and V are orthogonal
     * matrices and Sigma is a diagonal matrix.
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() const;

    /**
     * @brief Calculates the eigenvalues of the matrix.
     *
     * Eigenvalues are scalar values associated with a matrix that describe the
     * scaling factor of the eigenvectors when the matrix is multiplied by them.
     * For a square matrix A, if there exists a vector v and a scalar lambda
     * such that Av = lambda * v, then lambda is an eigenvalue and v is the
     * corresponding eigenvector. This method computes the eigenvalues using
     * the power iteration method.
     *
     * @return A vector containing the eigenvalues of the matrix.
     */
    std::vector<std::complex<T>> eigenvalues() const;

    /**
     * @brief Calculates the eigenvectors of the matrix.
     *
     * Eigenvectors are non-zero vectors that, when the matrix is multiplied by
     * them, result in a scalar multiple of themselves. For a square matrix A,
     * if there exists a vector v and a scalar lambda such that Av = lambda * v,
     * then v is an eigenvector corresponding to the eigenvalue lambda. This
     * method computes the eigenvectors using the power iteration method.
     *
     * @return A matrix where each column represents an eigenvector of the
     * matrix.
     */
    Matrix<T> eigenvectors() const;

    /**
     * @brief Calculates the rank of the matrix.
     *
     * The rank of a matrix is the maximum number of linearly independent rows
     * or columns in the matrix. It represents the dimension of the vector space
     * spanned by the rows or columns of the matrix. The rank is computed by
     * performing singular value decomposition (SVD) and counting the number of
     * non-zero singular values.
     *
     * @return The rank of the matrix.
     */
    size_t rank() const;

    /**
     * @brief Solves a system of linear equations Ax = b.
     *
     * @param b The right-hand side vector of the system.
     * @return The solution vector x.
     */
    Vector<T> solve(const Vector<T>& b) const;
};

}  // namespace cramer
