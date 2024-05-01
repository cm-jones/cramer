/*
 * This file is part of lincpp.
 *
 * lincpp is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * lincpp is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * lincpp. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __LINCPP_MATRIX_H__
#define __LINCPP_MATRIX_H__

#include <vector>
#include <complex>

namespace lincpp {

/**
 * @brief Represents a matrix of elements of type T.
 *
 * @tparam T The type of elements stored in the matrix.
 */
template <typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data; /**< The underlying container storing the matrix elements. */

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
    size_t rows() const;

    /**
     * @brief Returns the number of columns in the matrix.
     *
     * @return The number of columns in the matrix.
     */
    size_t cols() const;

    /**
     * @brief Generates an identity matrix of a specified size.
     * 
     * @param size The size of the identity matrix.
     * @return An identity matrix of the specified size.
     */
    Matrix<T> identity(size_t size);

    /**
     * @brief Overloads the () operator to access elements of the matrix.
     *
     * @param row The row index of the element to access.
     * @param col The column index of the element to access.
     * @return A reference to the element at the specified row and column.
     */
    T& operator()(size_t row, size_t col);

    /**
     * @brief Overloads the () operator to access elements of the matrix (const version).
     *
     * @param row The row index of the element to access.
     * @param col The column index of the element to access.
     * @return A const reference to the element at the specified row and column.
     */
    const T& operator()(size_t row, size_t col) const;

    /**
     * @brief Overloads the == operator to compare two matrices for equality.
     * 
     * @param other The matrix to compare with.
     * @return True if the matrices are equal, false otherwise.
     */
    bool operator==(const Matrix<T>& other) const;

    /**
     * @brief Overloads the + operator to perform matrix addition.
     *
     * @param other The matrix to add to the current matrix.
     * @return The result of matrix addition.
     */
    Matrix<T> operator+(const Matrix<T>& other) const;

    /**
     * @brief Overloads the += operator to perform matrix addition and assignment.
     *
     * @param other The matrix to add to the current matrix.
     * @return A reference to the updated matrix.
     */
    Matrix<T>& operator+=(const Matrix<T>& other);

    /**
     * @brief Overloads the - operator to perform matrix subtraction.
     *
     * @param other The matrix to subtract from the current matrix.
     * @return The result of matrix subtraction.
     */
    Matrix<T> operator-(const Matrix<T>& other) const;

    /**
     * @brief Overloads the -= operator to perform matrix subtraction and assignment.
     *
     * @param other The matrix to subtract from the current matrix.
     * @return A reference to the updated matrix.
     */
    Matrix<T>& operator-=(const Matrix<T>& other);

    /**
     * @brief Overloads the * operator to perform scalar multiplication.
     *
     * @param scalar The scalar value to multiply the matrix by.
     * @return The result of scalar multiplication.
     */
    Matrix<T> operator*(const T& scalar) const;

    /**
     * @brief Overloads the *= operator to perform scalar multiplication and assignment.
     *
     * @param scalar The scalar value to multiply the matrix by.
     * @return A reference to the updated matrix.
     */
    Matrix<T>& operator*=(const T& scalar);

    /**
     * @brief Overloads the * operator to perform matrix multiplication.
     *
     * @param other The matrix to multiply with.
     * @return The result of matrix multiplication.
     */
    Matrix<T> operator*(const Matrix<T>& other) const;

    /* Matrix properties */

    /**
     * @brief Determines if the matrix is diagonal.
     * 
     * @return True if the matrix is diagonal, false otherwise.
     */
    bool is_diagonal() const;

    /**
     * @brief Determines if the matrix is symmetric.
     * 
     * @return True if the matrix is symmetric, false otherwise.
     */
    bool is_symmetric() const;

    /**
     * @brief Determines if the matrix is Hermitian.
     * 
     * @return True if the matrix is Hermitian, false otherwise.
     */
    bool is_hermitian() const;

    /**
     * @brief Determines if a matrix is orthogonal.
     * 
     * @return True if the matrix is orthogonal, false otherwise.
     */

    bool is_orthogonal() const;

    /* Matrix mappings */

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
     */
    T det() const;

    /**
     * @brief Calculates the determinant of the matrix using LU decomposition.
     *
     * @return The determinant of the matrix.
     */
    T det_via_lu() const;

    /* Matrix transformations */

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
     */
    Matrix<T> inverse() const;

    /**
     * @brief Calculates the adjoint of the matrix.
     * 
     * @return The adjoint of the matrix.
     */
    Matrix<T> adjoint() const;

    /**
     * @brief Calculates the complex conjugate of the matrix.
     * 
     * @return
     */
    Matrix<T> conjugate() const;

    /**
     * @brief Calculates the matrix exponential of the matrix.
     *
     * @return The matrix exponential of the matrix.
     */
    Matrix<T> exp() const;

    /* Decompositions */

    /**
     * @brief Performs LU decomposition on the matrix.
     *
     * @return A pair of matrices (L, U) where L is a lower triangular matrix and U is an upper triangular matrix.
     */
    std::pair<Matrix<T>, Matrix<T>> lu() const;

    /**
     * @brief Performs QR decomposition on the matrix.
     * 
     * @return A pair of matrices (Q, R) where Q is an orthogonal matrix and R is an upper triangular matrix.
     */
    std::pair<Matrix<T>, Matrix<T>> qr() const;

    /**
     * @brief Performs SVD decomposition on the matrix.
     * 
     * @return A tuple of matrices (U, S, V) where U and V are orthogonal matrices and S is a diagonal matrix.
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() const;

    /* Eigenvalues and eigenvectors */

    /**
     * @brief Calculates the eigenvalues of the matrix.
     *
     * @return A vector containing the eigenvalues of the matrix.
     */
    std::vector<std::complex<T>> eigenvalues() const;

    /**
     * @brief Calculates the eigenvectors of the matrix.
     *
     * @return A matrix where each column represents an eigenvector of the matrix.
     */
    Matrix<T> eigenvectors() const;

    T Matrix<T>::max_norm() const
};

} // namespace lincpp

#endif // __LINCPP_MATRIX_H__
