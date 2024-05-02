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

#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

#include "../include/vector.h"
#include "../include/matrix.h"

namespace lincpp {

/* Constructors */

template <typename T>
Matrix<T>::Matrix() : rows(0), cols(0), data() {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows, std::vector<T>(cols)) {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T& value) : rows(rows), cols(cols), data(rows, std::vector<T>(cols, value)) {}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& values) : rows(values.size()), cols(values[0].size()), data(values) {}

template <typename T>
inline size_t Matrix<T>::rows() const {
    return rows;
}

template <typename T>
inline size_t Matrix<T>::cols() const {
    return cols;
}

template <typename T>
Matrix<T> Matrix<T>::identity(size_t size) {
    Matrix<T> I(size, size);

    for (size_t i = 0; i < size; ++i) {
        I(i, i) = static_cast<T>(1);
    }

    return I;
}

template <typename T>
Matrix<T> Matrix<T>::zeros(size_t rows, size_t cols) {
    return Matrix<T>(rows, cols, static_cast<T>(0));
}

template <typename T>
Matrix<T> Matrix<T>::ones(size_t rows, size_t cols) {
    return Matrix<T>(rows, cols, static_cast<T>(1));
}

template <typename T>
Matrix<T> Matrix<T>::random(size_t rows, size_t cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0, 1);

    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = dist(gen);
        }
    }
    return result;
}

/* Operators */

template <typename T>
inline T& Matrix<T>::operator()(size_t row, size_t col) {
    return data[row][col];
}

template <typename T>
inline const T& Matrix<T>::operator()(size_t row, size_t col) const {
    return data[row][col];
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        if (!std::equal(data[i].begin(), data[i].end(), other.data[i].begin())) {
            return false;
        }
    }

    return true;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must agree for addition.");
    }

    Matrix<T> sum(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum(i, j) = data[i][j] + other(i, j);
        }
    }

    return sum;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must agree for addition.");
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] += other(i, j);
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must agree for subtraction.");
    }

    Matrix<T> diff(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            diff(i, j) = data[i][j] - other(i, j);
        }
    }

    return diff;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must agree for subtraction.");
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] -= other(i, j);
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> product(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            product(i, j) = data[i][j] * scalar;
        }
    }

    return product;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] *= scalar;
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions must agree for multiplication.");
    }

    Matrix<T> product(rows, other.cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            T sum = static_cast<T>(0);
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other(k, j);
            }
            product(i, j) = sum;
        }
    }

    return product;
}

/* Matrix properties */

template <typename T>
bool Matrix<T>::is_diagonal() const {
    if (rows != cols) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (i != j && data[i][j] != static_cast<T>(0)) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool Matrix<T>::is_symmetric() const {
    if (rows != cols) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            if (data[i][j] != data[j][i]) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool Matrix<T>::is_hermitian() const {
    if (rows != cols) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            if (data[i][j] != std::conj(data[j][i])) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool Matrix<T>::is_orthogonal() const {
    if (rows != cols) {
        return false;
    }

    Matrix<T> product = (*this) * transpose();

    return product == identity(rows);
}

/* Matrix mappings */

template <typename T>
inline T Matrix<T>::trace() const {
    T trace = static_cast<T>(0);

    for (size_t i = 0; i < std::min(rows, cols); ++i) {
        trace += data[i][i];
    }

    return trace;
}

template <typename T>
T Matrix<T>::det() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate the determinant.");
    }

    if (rows == 1) {
        return data[0][0];
    }

    if (rows == 2) {
        return data[0][0] * data[1][1] - data[0][1] * data[1][0];
    }

    T det = static_cast<T>(0);

    for (size_t j = 0; j < cols; ++j) {
        Matrix<T> submatrix(rows - 1, cols - 1);

        for (size_t i = 1; i < rows; ++i) {
            size_t sub_j = 0;
            for (size_t k = 0; k < cols; ++k) {
                if (k == j) {
                    continue;
                }
                submatrix(i - 1, sub_j++) = data[i][k];
            }
        }

        det += (j % 2 == 0 ? 1 : -1) * data[0][j] * submatrix.det();
    }

    return det;
}

template <typename T>
T Matrix<T>::det_via_lu() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate the determinant via LU decomposition.");
    }

    auto [L, U] = lu();

    T det = static_cast<T>(1);

    for (size_t i = 0; i < rows; ++i) {
        det *= U(i, i);
    }

    return det;
}

/* Matrix transformations */

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> transpose(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transpose(j, i) = data[i][j];
        }
    }

    return transpose;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate the inverse.");
    }

    T det = this->det();

    if (det == static_cast<T>(0)) {
        throw std::invalid_argument("Matrix must be non-singular to calculate the inverse.");
    }

    Matrix<T> adj = adjoint();

    return adj * (static_cast<T>(1) / det);
}

template <typename T>
Matrix<T> Matrix<T>::adjoint() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate the adjoint.");
    }

    Matrix<T> adj(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            Matrix<T> submatrix(rows - 1, cols - 1);

            for (size_t k = 0; k < rows; ++k) {
                if (k == i) {
                    continue;
                }

                size_t sub_i = k < i ? k : k - 1;
                for (size_t l = 0; l < cols; ++l) {
                    if (l == j) {
                        continue;
                    }

                    size_t sub_j = l < j ? l : l - 1;
                    submatrix(sub_i, sub_j) = data[k][l];
                }
            }

            adj(j, i) = ((i + j) % 2 == 0 ? 1 : -1) * submatrix.det();
        }
    }

    return adj;
}

template <typename T>
Matrix<T> Matrix<T>::conjugate() const {
    Matrix<T> conjugate(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            conjugate(i, j) = std::conj(data[i][j]);
        }
    }

    return conjugate;
}

template <typename T>
Matrix<T> Matrix<T>::exp() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate the exponential.");
    }

    const int q = 6;
    const T norm = max_norm();
    const int s = std::max(0, static_cast<int>(std::ceil(std::log2(norm / q))));
    const T scale = std::pow(static_cast<T>(2), -s);

    Matrix<T> A = *this * scale;
    Matrix<T> P = identity(rows);
    Matrix<T> Q = identity(rows);

    const int num_terms = q;
    T c = static_cast<T>(1);

    for (int k = 1; k <= num_terms; ++k) {
        c *= static_cast<T>(q - k + 1) / static_cast<T>(k * (2 * q - k + 1));
        Q = A * Q;
        P += c * Q;
    }

    for (int k = 0; k < s; ++k) {
        P = P * P;
    }

    return P;
}

/* Decompositions */

template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::lu() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to perform LU decomposition.");
    }

    Matrix<T> L(rows, cols);
    Matrix<T> U(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (i <= j) {
                U(i, j) = data[i][j];

                for (size_t k = 0; k < i; ++k) {
                    U(i, j) -= L(i, k) * U(k, j);
                }

                if (i == j) {
                    L(i, j) = static_cast<T>(1);
                }
            } else {
                L(i, j) = data[i][j];

                for (size_t k = 0; k < j; ++k) {
                    L(i, j) -= L(i, k) * U(k, j);
                }

                L(i, j) /= U(j, j);
            }
        }
    }

    return std::make_pair(L, U);
}

template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::qr() const {
    Matrix<T> Q = identity(rows);
    Matrix<T> R = *this;

    for (size_t i = 0; i < std::min(rows, cols); ++i) {
        T norm = static_cast<T>(0);
        for (size_t j = i; j < rows; ++j) {
            norm += R(j, i) * R(j, i);
        }
        norm = std::sqrt(norm);

        T s = R(i, i) < static_cast<T>(0) ? norm : -norm;
        T alpha = static_cast<T>(1) / std::sqrt(static_cast<T>(2) * norm * (norm - R(i, i)));

        Vector<T> v(rows);
        v[i] = R(i, i) - s;
        for (size_t j = i + 1; j < rows; ++j) {
            v[j] = R(j, i);
        }

        R -= (v * v.transpose()) * (static_cast<T>(2) * alpha);
        Q -= (Q * v) * (v.transpose() * Q) * (static_cast<T>(2) * alpha);
    }

    return std::make_pair(Q, R);
}

template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> Matrix<T>::svd() const {
    Matrix<T> A = *this;
    Matrix<T> U = identity(rows);
    Matrix<T> V = identity(cols);

    const size_t max_iterations = 100;
    const T tolerance = std::numeric_limits<T>::epsilon();

    for (size_t i = 0; i < max_iterations; ++i) {
        auto [Q, R] = A.qr();
        A = R * Q;
        U *= Q;
        V *= Q;

        if (A.max_norm() < tolerance) {
            break;
        }
    }

    return std::make_tuple(U, A, V);
}

/* Eigenvalues and eigenvectors */

template <typename T>
std::vector<std::complex<T>> Matrix<T>::eigenvalues() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate its eigenvalues.");
    }
    if (det() == static_cast<T>(0)) {
        throw std::invalid_argument("Matrix must be non-singular to calculate its eigenvalues.");
    }

    const size_t max_iterations = 100;
    const T tolerance = std::numeric_limits<T>::epsilon();

    std::vector<std::complex<T>> eigvals;
    Matrix<T> A = *this;

    for (size_t i = 0; i < rows; ++i) {
        Matrix<T> x(rows, 1);
        x(0, 0) = static_cast<T>(1);

        for (size_t j = 0; j < max_iterations; ++j) {
            Matrix<T> y = A * x;
            x = y * (static_cast<T>(1) / y.max_norm());

            Matrix<T> z = A * x;
            T eigenvalue = (z.transpose() * x)(0, 0) / (x.transpose() * x)(0, 0);

            if ((z - x * eigenvalue).max_norm() < tolerance) {
                break;
            }
        }

        eigvals.emplace_back(A(i, i), static_cast<T>(0));

        for (size_t j = i; j < rows; ++j) {
            A(j, i) = A(i, j) = static_cast<T>(0);
        }

        A(i, i) = static_cast<T>(1);
    }

    return eigvals;
}

template <typename T>
Matrix<T> Matrix<T>::eigenvectors() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate its eigenvectors.");
    }
    if (det() == static_cast<T>(0)) {
        throw std::invalid_argument("Matrix must be non-singular to calculate its eigenvectors.");
    }

    const size_t max_iterations = 100;
    const T tolerance = std::numeric_limits<T>::epsilon();

    Matrix<T> eigvecs(rows, cols);
    Matrix<T> A = *this;

    for (size_t i = 0; i < rows; ++i) {
        Matrix<T> x(rows, 1);
        x(0, 0) = static_cast<T>(1);

        for (size_t j = 0; j < max_iterations; ++j) {
            Matrix<T> y = A * x;
            x = y * (static_cast<T>(1) / y.max_norm());
        }

        for (size_t j = 0; j < rows; ++j) {
            eigvecs(j, i) = x(j, 0);
        }

        for (size_t j = i; j < rows; ++j) {
            A(j, i) = A(i, j) = static_cast<T>(0);
        }

        A(i, i) = static_cast<T>(1);
    }

    return eigvecs;
}

template <typename T>
size_t Matrix<T>::rank() const {
    auto [U, S, V] = svd();
    const T tolerance = std::numeric_limits<T>::epsilon() * std::max(rows, cols) * S(0, 0);
    size_t rank = 0;

    for (size_t i = 0; i < std::min(rows, cols); ++i) {
        if (S(i, i) > tolerance) {
            ++rank;
        }
    }

    return rank;
}

/* Helper functions */

template <typename T>
inline T Matrix<T>::max_norm() const {
    T norm = static_cast<T>(0);

    for (size_t i = 0; i < rows; ++i) {
        T row_sum = static_cast<T>(0);
        for (size_t j = 0; j < cols; ++j) {
            row_sum += std::abs(data[i][j]);
        }
        norm = std::max(norm, row_sum);
    }

    return norm;
}

} // namespace lincpp
