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
    T norm = static_cast<T>(0);

    for (size_t i = 0; i < rows; ++i) {
        T row_sum = static_cast<T>(0);
        for (size_t j = 0; j < cols; ++j) {
            row_sum += std::abs(data[i][j]);
        }
        norm = (std::abs(row_sum) > std::abs(norm)) ? row_sum : norm;
    }

    return norm;
}

// Constructors

template <typename T>
Matrix<T>::Matrix() : rows(0), cols(0), data() {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols), data(rows, std::vector<T>(cols)) {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T& value)
    : rows(rows), cols(cols), data(rows, std::vector<T>(cols, value)) {}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& values)
    : rows(values.size()), cols(values[0].size()), data(values) {}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, std::initializer_list<T> values)
    : rows(rows), cols(cols), data(rows, std::vector<T>(cols)) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument(
            "Initializer list size does not match matrix dimensions");
    }
    auto iter = values.begin();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = *iter++;
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
    Matrix<T> result(size, size);

    for (size_t i = 0; i < size; ++i) {
        result(i, i) = static_cast<T>(1);
    }

    return result;
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
    static_assert(std::is_floating_point<T>::value ||
                      std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, std::complex<double>>::value,
                  "random() is only supported for floating-point types and "
                  "complex types");

    std::random_device rand;
    std::mt19937 gen(rand());

    if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dis(0, 1);
        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = dis(gen);
            }
        }
        return result;
    } else if constexpr (std::is_same<T, std::complex<float>>::value ||
                         std::is_same<T, std::complex<double>>::value) {
        using value_type = typename T::value_type;
        std::uniform_real_distribution<value_type> dis(0, 1);
        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = T(dis(gen), dis(gen));
            }
        }
        return result;
    }
}

// Operators

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
        if (!std::equal(data[i].begin(), data[i].end(),
                        other.data[i].begin())) {
            return false;
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
            sum(i, j) = data[i][j] + other(i, j);
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
            data[i][j] += other(i, j);
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
        throw std::invalid_argument(
            "Matrix dimensions must agree for subtraction.");
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

    // Calculate the optimum block size based on cache parameters and matrix
    // sizes
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
                            sum += data[ii][kk] * other(kk, jj);
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

// Matrix properties

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
            if (i != j && data[i][j] != static_cast<T>(0)) {
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
            if (data[i][j] != data[j][i]) {
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
        for (size_t j = i + 1; j < cols; ++j) {
            if (std::real(data[i][j]) != std::real(std::conj(data[j][i])) ||
                std::imag(data[i][j]) != -std::imag(std::conj(data[j][i]))) {
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
inline T Matrix<T>::trace() const {
    T trace = static_cast<T>(0);

    for (size_t i = 0; i < std::min(rows, cols); ++i) {
        trace += data[i][i];
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
        return data[0][0];
    }

    // If the matrix is 2x2, use the formula ad - bc
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

        T factor = (j % 2 == 0) ? static_cast<T>(1) : static_cast<T>(-1);
        det += factor * data[0][j] * submatrix.det();
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
            transpose(j, i) = data[i][j];
        }
    }

    return transpose;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate the inverse.");
    }

    T det = this->det();

    if constexpr (std::is_floating_point<T>::value) {
        if (std::abs(det) < std::numeric_limits<T>::epsilon()) {
            throw std::invalid_argument(
                "Matrix must be non-singular to calculate the inverse.");
        }
    } else {
        if (det == static_cast<T>(0)) {
            throw std::invalid_argument(
                "Matrix must be non-singular to calculate the inverse.");
        }
    }

    Matrix<T> adj = adjoint();

    return adj * (static_cast<T>(1) / det);
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
                    submatrix(sub_i, sub_j) = data[k][l];
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

    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = std::conj(data[i][j]);
            }
        }
    } else {
        // For non-complex types, conjugate is the same as the original value
        result = *this;
    }

    return result;
}

// Matrix functions

template <typename T>
Matrix<T> Matrix<T>::exp() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate the exponential.");
    }

    const int q = 6;
    const T norm = max_norm();
    const int s =
        std::max(0, static_cast<int>(std::ceil(std::log2(std::abs(norm) / q))));
    const T scale = std::pow(static_cast<T>(2), static_cast<T>(-s));

    Matrix<T> A = *this * scale;
    Matrix<T> P = identity(rows);
    Matrix<T> Q = identity(rows);

    T c = static_cast<T>(1);

    for (int k = 1; k <= q; ++k) {
        c *= static_cast<T>(q - k + 1) / static_cast<T>(k * (2 * q - k + 1));
        Q = A * Q;
        P = P + Q * c;  // Changed order of multiplication
    }

    for (int k = 0; k < s; ++k) {
        P = P * P;
    }

    return P;
}

template <typename T>
Matrix<T> Matrix<T>::pow(int n) const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate its integer power.");
    }

    if (n == 0) {
        return identity(rows);
    }

    Matrix<T> result = identity(rows);
    Matrix<T> base = *this;
    bool negative = (n < 0);
    n = std::abs(n);

    while (n > 0) {
        if (n % 2 == 1) {
            result = result * base;  // Use matrix multiplication instead of *=
        }
        base = base * base;  // Use matrix multiplication instead of *=
        n /= 2;
    }

    return negative ? result.inverse() : result;
}

template <typename T>
Matrix<T> Matrix<T>::sqrt() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate its square root.");
    }

    // This is a placeholder implementation using Denman-Beavers iteration
    // For a more robust implementation, consider using Schur decomposition or
    // other methods
    Matrix<T> X = *this;
    Matrix<T> Y = identity(rows);
    const int max_iterations = 20;
    const T tolerance = std::numeric_limits<T>::epsilon();

    for (int i = 0; i < max_iterations; ++i) {
        Matrix<T> X_next = (X + Y.inverse()) * static_cast<T>(0.5);
        Matrix<T> Y_next = (Y + X.inverse()) * static_cast<T>(0.5);

        if (std::abs((X_next - X).max_norm()) < std::abs(tolerance)) {
            return X_next;
        }

        X = X_next;
        Y = Y_next;
    }

    throw std::runtime_error("Square root iteration did not converge.");
}

template <typename T>
Matrix<T> Matrix<T>::log() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to calculate its logarithm.");
    }

    const int q = 6;

    Matrix<T> A = *this;
    int s = 0;

    while (std::abs(A.max_norm()) > 0.5) {
        A = A.sqrt();
        ++s;
    }

    Matrix<T> I = identity(rows);
    Matrix<T> Z = A - I;
    Matrix<T> X = Z;
    Matrix<T> P = I;
    Matrix<T> Q = I;

    for (int k = 1; k <= q; ++k) {
        P = P * (Z * (static_cast<T>(q - k + 1) /
                      static_cast<T>(k * (2 * q - k + 1))));
        Q = Q * (Z * (-static_cast<T>(q - k + 1) /
                      static_cast<T>(k * (2 * q - k + 1))));
        X = X + P + Q;
    }

    for (int k = 0; k < s; ++k) {
        X = X + X;
    }

    return X;
}

// Matrix decompositions

template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::lu() const {
    if (!is_square()) {
        throw std::invalid_argument(
            "Matrix must be square to perform LU decomposition.");
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
        Vector<T> v(rows);
        T norm = static_cast<T>(0);

        for (size_t j = i; j < rows; ++j) {
            v[j] = R(j, i);
            norm += v[j] * v[j];
        }

        norm = std::sqrt(norm);

        T s = (std::real(R(i, i)) < 0) ? norm : -norm;
        T alpha = static_cast<T>(1) /
                  std::sqrt(static_cast<T>(2) * norm * (norm - R(i, i)));

        v[i] = R(i, i) - s;

        Matrix<T> vvT(rows, cols);
        for (size_t j = 0; j < rows; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                vvT(j, k) = v[j] * v[k];
            }
        }

        R = R - vvT * (static_cast<T>(2) * alpha);
        Q = Q - (Q * vvT) * (static_cast<T>(2) * alpha);
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
        U = U * Q;
        V = V * Q;

        if (std::abs(A.max_norm()) < std::abs(tolerance)) {
            break;
        }
    }

    return std::make_tuple(U, A, V);
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
size_t Matrix<T>::rank() const {
    auto [U, S, V] = svd();
    const T tolerance = std::numeric_limits<T>::epsilon() *
                        static_cast<T>(std::max(rows, cols)) *
                        std::abs(S(0, 0));
    size_t rank = 0;

    for (size_t i = 0; i < std::min(rows, cols); ++i) {
        if (std::abs(S(i, i)) > std::abs(tolerance)) {
            ++rank;
        }
    }

    return rank;
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
