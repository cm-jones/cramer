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

#include "../include/vector.h"
#include "../include/matrix.h"

namespace numcpp {

template <typename T>
Matrix<T>::Matrix() {
    this->rows = 0;
    this->cols = 0;
    this->data = std::vector<std::vector<T>>();

    return *this;
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = std::vector<std::vector<T>>(rows, std::vector<T>(cols));

    return *this;
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T& value) {
    this->rows = rows;
    this->cols = cols;
    this->data = std::vector<std::vector<T>>(rows, std::vector<T>(cols, value));

    for (size_t i = 0; i < rows; ++i) {
        this->data[i] = std::vector<T>(cols, value);
    }

    return *this;
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& values) {
    this->rows = values.size();
    this->cols = values[0].size();
    this->data = values;

    return *this;
}

template <typename T>
size_t Matrix<T>::rows() const {
    if (this->data.empty()) {
        return 0;
    }
    
    if (this->data[0].empty()) {
        return 0;
    }
    
    return this->rows;
}

template <typename T>
size_t Matrix<T>::cols() const {
    if (this->data.empty()) {
        return 0;
    }

    if (this->data[0].empty()) {
        return 0;
    }

    return this->cols;
}

template <typename T>
T& Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= this->rows() || col >= this->cols())
        throw std::out_of_range("Matrix index out of bounds.");

    return this->data[row][col];
}

template <typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= this->rows() || col >= this->cols())
        throw std::out_of_range("Matrix index out of bounds.");
    
    return this->data[row][col];
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero for transposition.");
    }
    
    if (this->rows == 1 && this->cols == 1) {
        return *this;
    }

    Matrix<T> result(this->cols, this->rows);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result(j, i) = this->data[i][j];
        }
    }

    return result;
}

template <typename T>
T Matrix<T>::trace() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the trace.");
    
    T trace = 0;

    for (size_t i = 0; i < this->rows; ++i) {
        trace += this->data[i][i];
    }

    return trace;
}

template <typename T>
T Matrix<T>::determinant() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the determinant.");
    
    T det = 0;

    for (size_t i = 0; i < this->rows; ++i) {
        T subdet = 0;
        for (size_t j = 0; j < this->cols; ++j) {
            subdet += this->data[i][j] * this->data[i + 1][j + 1] - this->data[i][j + 1] * this->data[i + 1][j];
        }
        det += subdet;
    }

    return det;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the inverse.");
    
    if (this->determinant() == 0)
        throw std::invalid_argument("Matrix must be non-singular to calculate the inverse.");

    
}

template <typename T>
std::vector<std::complex<T>> Matrix<T>::eigenvalues() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the eigenvalues.");
    
    if (this->determinant() == 0)
        throw std::invalid_argument("Matrix must be non-singular to calculate the eigenvalues.");
    
    
}

template <typename T>
std::vector<Matrix<T>> Matrix<T>::eigenvectors() const {
    // Implementation goes here
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (this->rows == 0 || this->cols == 0 || other.rows == 0 || other.cols == 0)
        throw std::invalid_argument("Matrix dimensions must be non-zero for multiplication.");

    if (this->cols != other.rows)
        throw std::invalid_argument("Matrix dimensions must agree for multiplication.");

    Matrix<T> result(this->rows, other.cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < this->cols; ++k) {
                result(i, j) += this->data[i][k] * other(k, j);
            }
        }
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    if (this->rows == 0 || this->cols == 0)
        throw std::invalid_argument("Matrix dimensions must be non-zero for multiplication.");
    
    Matrix<T> result(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result(i, j) = this->data[i][j] * scalar;
        }
    }

    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar) {
    if (this->rows == 0 || this->cols == 0)
        throw std::invalid_argument("Matrix dimensions must be non-zero for multiplication.");
    
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            this->data[i][j] *= scalar;
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must agree for addition.");

    Matrix<T> result(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result(i, j) = this->data[i][j] + other(i, j);
        }
    }

    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
    // Implementation goes here
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    // Implementation goes here
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
    // Implementation goes here
}

// Other matrix operations and methods
// ...

} // namespace numcpp
