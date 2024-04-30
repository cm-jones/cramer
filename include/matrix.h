/*
 * This file is part of libdsc.
 *
 * libdsc is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * libdsc is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * libdsc. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __NUMCPP_MATRIX_H__
#define __NUMCPP_MATRIX_H__

#include <vector>
#include <complex>

namespace numcpp {

/**
 * @brief Represents a matrix of elements of type T.
 *
 * @tparam T The type of elements stored in the matrix.
 */
template <typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data; /**< The underlying container storing the matrix elements. */
    size_t rows;
    size_t cols;

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
     * @brief Calculates the transpose of the matrix.
     *
     * @return The transpose of the matrix.
     */
    Matrix<T> transpose() const;

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
    T determinant() const;

    /**
     * @brief Calculates the inverse of the matrix.
     *
     * @return The inverse of the matrix.
     */
    Matrix<T> inverse() const;

    /**
     * @brief Calculates the eigenvalues of the matrix.
     *
     * @return A vector containing the eigenvalues of the matrix.
     */
    std::vector<std::complex<T>> eigenvalues() const;

    /**
     * @brief Calculates the eigenvectors of the matrix.
     *
     * @return A vector containing the eigenvectors of the matrix.
     */
    std::vector<Matrix<T>> eigenvectors() const;

    /**
     * @brief Overloads the * operator to perform matrix multiplication.
     *
     * @param other The matrix to multiply with.
     * @return The result of matrix multiplication.
     */
    Matrix<T> operator*(const Matrix<T>& other) const;

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

    // Other matrix operations and methods
    // ...
};

}  // namespace numcpp

#endif  // __NUMCPP_MATRIX_H__
