// SPDX-License-Identifier: GPL-3.0-or-later

#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <type_traits>

#include "vector.hpp"

namespace cramer {

// Helper methods

template <typename T>
T Matrix<T>::max_norm() const {
    T norm = T(0);

    for (size_t i = 0; i < rows; ++i) {
        T row_sum = T(0);

        for (size_t j = 0; j < cols; ++j) {
            if constexpr (std::is_arithmetic<T>::value) {
                row_sum += std::abs(entries[i][j]);
            } else if constexpr (std::is_same<T, std::complex<float>>::value ||
                                 std::is_same<T, std::complex<double>>::value) {
                row_sum += std::abs(entries[i][j]);
            }
        }

        if (std::abs(row_sum) > std::abs(norm)) {
            norm = row_sum;
        }
    }

    return norm;
}

// Constructors

template <typename T>
Matrix<T>::Matrix() : rows(0), cols(0), entries() {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols), entries(rows, std::vector<T>(cols)) {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T& value)
    : rows(rows), cols(cols), entries(rows, std::vector<T>(cols, value)) {}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& values)
    : rows(values.size()), cols(values[0].size()), entries(values) {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, std::initializer_list<T> values)
    : rows(rows), cols(cols), entries(rows, std::vector<T>(cols)) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument(
            "Initializer list size does not match matrix dimensions");
    }

    auto iter = values.begin();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            entries[i][j] = *iter++;
        }
    }
}

// Getters

template <typename T>
inline size_t Matrix<T>::get_rows() const {
    return rows;
}

template <typename T>
inline size_t Matrix<T>::get_cols() const {
    return cols;
}

// Special matrices

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

// Operators

template <typename T>
inline T& Matrix<T>::operator()(size_t row, size_t col) {
    return entries[row][col];
}

template <typename T>
inline const T& Matrix<T>::operator()(size_t row, size_t col) const {
    return entries[row][col];
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (entries[i][j] != other(i, j)) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Matrix dimensions must agree for addition.");
    }

    Matrix<T> sum(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum(i, j) = entries[i][j] + other(i, j);
        }
    }

    return sum;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Matrix dimensions must agree for addition.");
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            entries[i][j] += other(i, j);
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Matrix dimensions must agree for subtraction.");
    }

    Matrix<T> difference(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            difference(i, j) = entries[i][j] - other(i, j);
        }
    }

    return difference;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Matrix dimensions must agree for subtraction.");
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            entries[i][j] -= other(i, j);
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix<T> product(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            product(i, j) = entries[i][j] * scalar;
        }
    }

    return product;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            entries[i][j] *= scalar;
        }
    }

    return *this;
}

// Matrix multiplication using cache blocking

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument(
            "Matrix dimensions must agree for multiplication.");
    }

    Matrix<T> product(rows, other.cols);

    // Cache parameters
    const long CACHE_LINE_SIZE = 64;     // Typical cache line size
    const long CACHE_SIZE = 32768;       // 32 KB L1 cache
    const long CACHE_ASSOCIATIVITY = 8;  // 8-way set associative

    // Get optimum block size based on cache parameters and matrix sizes
    const size_t BLOCK_SIZE =
        std::min(static_cast<size_t>(
                     std::sqrt(CACHE_SIZE / CACHE_ASSOCIATIVITY / sizeof(T))),
                 static_cast<size_t>(CACHE_LINE_SIZE / sizeof(T)));

    // Perform cache blocking matrix multiplication
    for (size_t i = 0; i < rows; i += BLOCK_SIZE) {
        for (size_t j = 0; j < other.cols; j += BLOCK_SIZE) {
            for (size_t k = 0; k < cols; k += BLOCK_SIZE) {
                // Multiply the current blocks
                for (size_t ii = i; ii < std::min(i + BLOCK_SIZE, rows); ++ii) {
                    for (size_t jj = j;
                         jj < std::min(j + BLOCK_SIZE, other.cols); ++jj) {
                        T sum = static_cast<T>(0);
                        for (size_t kk = k; kk < std::min(k + BLOCK_SIZE, cols);
                             ++kk) {
                            sum += entries[ii][kk] * other(kk, jj);
                        }
                        product(ii, jj) += sum;
                    }
                }
            }
        }
    }

    return product;
}

template <typename T>
Vector<T> Matrix<T>::multiply_vector(const Vector<T>& vec) const {
    if (this->get_cols() != vec.size()) {
        throw std::invalid_argument(
            "Matrix and vector dimensions do not match for multiplication");
    }
    Vector<T> result(this->get_rows());
    for (size_t row = 0; row < this->get_rows(); ++row) {
        T sum = T();
        for (size_t col = 0; col < this->get_cols(); ++col) {
            sum += (*this)(row, col) * vec[col];
        }
        result[row] = sum;
    }
    return result;
}

// Matrix properties/tests

template <typename T>
bool Matrix<T>::is_square() const {
    return rows == cols;
}

template <typename T>
bool Matrix<T>::is_diagonal() const {
    if (!is_square()) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (i != j && entries[i][j] != static_cast<T>(0)) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool Matrix<T>::is_symmetric() const {
    if (!is_square()) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            if (entries[i][j] != entries[j][i]) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool Matrix<T>::is_invertible() const {
    if (!is_square()) {
        return false;
    }

    if constexpr (std::is_floating_point<T>::value) {
        return std::abs(det()) > std::numeric_limits<T>::epsilon();
    } else {
        return det() != static_cast<T>(0);
    }
}

template <typename T>
bool Matrix<T>::is_hermitian() const {
    if (!is_square()) {
        return false;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = i; j < cols; ++j) {
            if (i == j) {
                if (std::imag(entries[i][j]) != 0) {
                    return false;
                }
            } else if (entries[i][j] != std::conj(entries[j][i])) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool Matrix<T>::is_orthogonal() const {
    if (!is_square()) {
        return false;
    }

    Matrix<T> product = (*this) * transpose();

    return product == identity(rows);
}

template <typename T>
bool Matrix<T>::is_unitary() const {
    if (!is_square()) {
        return false;
    }

    Matrix<T> product = (*this) * conjugate();

    return product == identity(rows);
}

template <typename T>
inline T Matrix<T>::trace() const {
    T trace = static_cast<T>(0);

    for (size_t i = 0; i < std::min(rows, cols); ++i) {
        trace += entries[i][i];
    }

    return trace;
}

template <typename T>
T Matrix<T>::det() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate the determinant.");
    }

    // If the matrix is 1x1, return the single element
    if (rows == 1) {
        return entries[0][0];
    }

    // If the matrix is 2x2, use the formula ad - bc
    if (rows == 2) {
        return entries[0][0] * entries[1][1] - entries[0][1] * entries[1][0];
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

                submatrix(i - 1, sub_j++) = entries[i][k];
            }
        }

        T factor = (j % 2 == 0) ? static_cast<T>(1) : static_cast<T>(-1);
        det += factor * entries[0][j] * submatrix.det();
    }

    return det;
}

template <typename T>
T Matrix<T>::det_via_lu() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate the determinant via LU "
            "decomposition.");
    }

    auto [L, U] = lu();

    T det = static_cast<T>(1);

    for (size_t i = 0; i < rows; ++i) {
        det *= U(i, i);
    }

    return det;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> transpose(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transpose(j, i) = entries[i][j];
        }
    }

    return transpose;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (!is_square()) {
        throw std::invalid_argument("Matrix must be square to calculate the inverse.");
    }

    T determinant = this->det();

    // Use std::abs for both real and complex numbers
    if (std::abs(determinant) < std::numeric_limits<typename std::remove_cv<decltype(std::abs(determinant))>::type>::epsilon()) {
        throw std::invalid_argument("Matrix must be non-singular to calculate the inverse.");
    }

    Matrix<T> adj = this->adjoint();

    return adj * (static_cast<T>(1) / determinant);
}

template <typename T>
Matrix<T> Matrix<T>::adjoint() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate the adjoint.");
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
                    submatrix(sub_i, sub_j) = entries[k][l];
                }
            }

            adj(j, i) =
                static_cast<T>(((i + j) % 2 == 0 ? 1 : -1)) * submatrix.det();
        }
    }

    return adj;
}

template <typename T>
Matrix<T> Matrix<T>::conjugate() const {
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if constexpr (std::is_same_v<T, std::complex<float>> ||
                          std::is_same_v<T, std::complex<double>>) {
                result(i, j) = std::conj(entries[i][j]);
            } else {
                result(i, j) = entries[i][j];
            }
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::outer(const Vector<T>& u, const Vector<T>& v) {
    Matrix<T> result(u.size(), v.size());

    for (size_t i = 0; i < u.size(); ++i) {
        for (size_t j = 0; j < v.size(); ++j) {
            result(i, j) = u[i] * v[j];
        }
    }

    return result;
}

// Matrix decompositions/factorizations

template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::lu() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to perform LU decomposition.");
    }

    Matrix<T> lower(rows, cols);
    Matrix<T> upper(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (i <= j) {
                upper(i, j) = entries[i][j];

                for (size_t k = 0; k < i; ++k) {
                    upper(i, j) -= lower(i, k) * upper(k, j);
                }

                if (i == j) {
                    lower(i, j) = static_cast<T>(1);
                }
            } else {
                lower(i, j) = entries[i][j];

                for (size_t k = 0; k < j; ++k) {
                    lower(i, j) -= lower(i, k) * upper(k, j);
                }

                lower(i, j) /= upper(j, j);
            }
        }
    }

    return std::make_pair(lower, upper);
}

template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::qr() const {
    size_t m = rows;
    size_t n = cols;
    Matrix<T> q(m, n);
    Matrix<T> r(n, n);

    for (size_t j = 0; j < n; ++j) {
        Vector<T> v(m);
        for (size_t i = 0; i < m; ++i) {
            v[i] = (*this)(i, j);
        }

        for (size_t i = 0; i < j; ++i) {
            Vector<T> q_i(m);
            for (size_t k = 0; k < m; ++k) {
                q_i[k] = q(k, i);
            }
            r(i, j) = q_i.dot(v);
            v -= q_i * r(i, j);
        }

        r(j, j) = v.norm();
        if (std::abs(r(j, j)) > std::numeric_limits<typename std::remove_cv<decltype(std::abs(r(j, j)))>::type>::epsilon()) {
            T scale = static_cast<T>(1) / r(j, j);
            for (size_t i = 0; i < m; ++i) {
                q(i, j) = v[i] * scale;
            }
        }
    }

    return std::make_pair(q, r);
}

template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> Matrix<T>::svd() const {
    const size_t row_count = rows;
    const size_t col_count = cols;
    const size_t sing_val_count = std::min(row_count, col_count);

    const size_t max_iterations = 100;
    const T convergence_threshold = 1e-6;
    const T initial_vector_value = 1.0;

    Matrix<T> left_sing_vecs(row_count, sing_val_count);
    Matrix<T> sing_vals(sing_val_count, sing_val_count, 0.0);
    Matrix<T> right_sing_vecs(col_count, sing_val_count);

    Matrix<T> work_matrix = *this;

    for (size_t idx = 0; idx < sing_val_count; ++idx) {
        Vector<T> right_vec(col_count, initial_vector_value);

        // Power iteration
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            Vector<T> left_vec = multiply_vector(right_vec);
            T sigma = left_vec.norm();
            left_vec = left_vec * (static_cast<T>(1) / sigma);

            Vector<T> new_right_vec = transpose().multiply_vector(left_vec);
            T new_sigma = new_right_vec.norm();
            new_right_vec = new_right_vec * (static_cast<T>(1) / new_sigma);

            // Modified comparison for complex numbers
            if (std::abs((right_vec - new_right_vec).norm()) <
                std::abs(static_cast<T>(convergence_threshold))) {
                break;
            }

            right_vec = new_right_vec;
        }

        Vector<T> left_vec = multiply_vector(right_vec);
        T sigma = left_vec.norm();
        left_vec = left_vec * (static_cast<T>(1) / sigma);

        // Store the singular values and vectors
        for (size_t row = 0; row < row_count; ++row) {
            left_sing_vecs(row, idx) = left_vec[row];
        }

        for (size_t col = 0; col < col_count; ++col) {
            right_sing_vecs(col, idx) = right_vec[col];
        }

        sing_vals(idx, idx) = sigma;

        // Deflate the matrix
        work_matrix = work_matrix - outer(left_vec, right_vec) * sigma;
    }

    return std::make_tuple(left_sing_vecs, sing_vals,
                           right_sing_vecs.transpose());
}

// Eigenvalues and eigenvectors

template <typename T>
std::vector<std::complex<T>> Matrix<T>::eigenvalues() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate its eigenvalues.");
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
            T eigenvalue =
                (z.transpose() * x)(0, 0) / (x.transpose() * x)(0, 0);

            if (std::abs((z - x * eigenvalue).max_norm()) <
                std::abs(tolerance)) {
                break;
            }
        }

        eigvals.emplace_back(A(i, i));

        for (size_t j = i; j < rows; ++j) {
            A(j, i) = A(i, j) = static_cast<T>(0);
        }

        A(i, i) = static_cast<T>(1);
    }

    return eigvals;
}

template <typename T>
Matrix<T> Matrix<T>::eigenvectors() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate its eigenvectors.");
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
Vector<T> Matrix<T>::solve(const Vector<T>& b) const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to solve a linear system.");
    }

    if (rows != b.size()) {
        throw std::invalid_argument(
            "Matrix and vector dimensions must agree for solving a linear "
            "system.");
    }

    auto [L, U] = lu();

    // Forward substitution
    Vector<T> y(rows);
    for (size_t i = 0; i < rows; ++i) {
        y[i] = b[i];
        for (size_t j = 0; j < i; ++j) {
            y[i] -= L(i, j) * y[j];
        }
        y[i] /= L(i, i);
    }

    // Backward substitution
    Vector<T> x(rows);
    for (size_t i = rows; i > 0; --i) {
        x[i - 1] = y[i - 1];
        for (size_t j = i; j < rows; ++j) {
            x[i - 1] -= U(i - 1, j) * x[j];
        }
        x[i - 1] /= U(i - 1, i - 1);
    }

    return x;
}

// Explicit template instantiations
template class cramer::Matrix<float>;
template class cramer::Matrix<double>;
template class cramer::Matrix<std::complex<float>>;
template class cramer::Matrix<std::complex<double>>;

}  // namespace cramer
