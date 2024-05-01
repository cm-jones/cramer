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

#include "../include/vector.h"
#include "../include/matrix.h"
#include <cmath>
#include <algorithm>

namespace lincpp {

/* Constructors */

template <typename T>
Matrix<T>::Matrix() {
    this->rows = 0;
    this->cols = 0;
    this->data = std::vector<std::vector<T>>();
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = std::vector<std::vector<T>>(rows, std::vector<T>(cols));
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T& value) {
    this->rows = rows;
    this->cols = cols;
    this->data = std::vector<std::vector<T>>(rows, std::vector<T>(cols, value));
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& values) {
    this->rows = values.size();
    this->cols = values[0].size();
    this->data = values;
}

template <typename T>
size_t Matrix<T>::rows() const {
    return data.size();
}

template <typename T>
size_t Matrix<T>::cols() const {
    return data.empty() ? 0 : data[0].size();
}

template <typename T>
Matrix<T> Matrix<T>::identity(size_t size) {
    Matrix<T> I(size, size);

    for (size_t i = 0; i < size; ++i) {
        I(i, i) = 1.0;
    }

    return I;
}

/* Operators */

template <typename T>
T& Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= this->rows() || col >= this->cols()) {
        throw std::out_of_range("Matrix index out of bounds.");
    }

    return this->data[row][col];
}

template <typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= this->rows() || col >= this->cols())
        throw std::out_of_range("Matrix index out of bounds.");
    
    return this->data[row][col];
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols)
        return false;

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            if (this->data[i][j] != other(i, j))
                return false;
        }
    }

    return true;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must agree for addition.");

    Matrix<T> sum(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            sum(i, j) = this->data[i][j] + other(i, j);
        }
    }

    return sum;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must agree for addition.");
    }

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            this->data[i][j] += other(i, j);
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (this->rows != other.rows || this->cols != other.cols)
        throw std::invalid_argument("Matrix dimensions must agree for subtraction.");

    Matrix<T> difference(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            difference(i, j) = this->data[i][j] - other(i, j);
        }
    }

    return difference;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must agree for subtraction.");
    }

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            this->data[i][j] -= other(i, j);
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero for multiplication.");
    }
    
    Matrix<T> product(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            product(i, j) = this->data[i][j] * scalar;
        }
    }

    return product;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar) {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero for multiplication.");
    }
    
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            this->data[i][j] *= scalar;
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (this->rows == 0 || this->cols == 0 || other.rows == 0 || other.cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero for multiplication.");
    }

    // If A is m x n, then B must be n x p for any p
    if (this->cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions must agree for multiplication.");
    }

    Matrix<T> product(this->rows, other.cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < this->cols; ++k) {
                product(i, j) += this->data[i][k] * other(k, j);
            }
        }
    }

    return product;
}

/* Matrix properties */

template <typename T>
bool Matrix<T>::is_diagonal() const {
    if (this->rows == 0 || this->cols == 0)
        throw std::invalid_argument("Matrix dimensions must be non-zero to check for diagonal.");

    // A matrix is diagonal if all off-diagonal elements are zero
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            if (i != j && this->data[i][j] != 0) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
bool Matrix<T>::is_symmetric() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to check for symmetry.");

    Matrix<T> transpose = this->transpose();

    // A matrix is symmetric if A = A^T
    return transpose == *this;
}

template <typename T>
bool Matrix<T>::is_hermitian() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to check for Hermitian symmetry.");

    Matrix<T> conjugate = this->conjugate();
    Matrix<T> transpose = this->transpose();

    // A matrix is Hermitian if A = A^H
    return conjugate == transpose;
}

template <typename T>
bool Matrix<T>::is_orthogonal() const {
    if (this->rows == 0 || this->cols == 0)
        throw std::invalid_argument("Matrix dimensions must be non-zero to check for orthogonality.");

    // A matrix is orthogonal if A^T * A = I
    Matrix<T> product = this->transpose() * (*this);
    Matrix<T> identity = Matrix<T>::identity(this->cols);

    return product == identity;
}

/* Matrix mappings */

template <typename T>
T Matrix<T>::trace() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the trace.");
    
    T trace = 0;

    // Compute the sum of the diagonal elements
    for (size_t i = 0; i < this->rows; ++i) {
        trace += this->data[i][i];
    }

    return trace;
}

template <typename T>
T Matrix<T>::det() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the determinant.");
    
    // If A is 1 x 1, then det(A) = A
    if (this->rows == 1)
        return this->data[0][0];
    
    // If A is 2 x 2, then det(A) = ad - bc
    if (this->rows == 2) {
        return this->data[0][0] * this->data[1][1] - this->data[0][1] * this->data[1][0];
    }
    
    T det = 0;

    // If A is n x n, then det(A) = sum of a_{0,j} * det(A_{0,j}) for j = 0 to n - 1
    for (size_t j = 0; j < this->cols; ++j) {
        Matrix<T> submatrix(this->rows - 1, this->cols - 1);

        for (size_t i = 1; i < this->rows; ++i) {
            for (size_t k = 0; k < this->cols; ++k) {
                if (k < j)
                    submatrix(i - 1, k) = this->data[i][k];
                else if (k > j)
                    submatrix(i - 1, k - 1) = this->data[i][k];
            }
        }

        det += (j % 2 == 0 ? 1 : -1) * this->data[0][j] * submatrix.det();
    }

    return det;
}

template <typename T>
T Matrix<T>::det_via_lu() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the determinant via LU decomposition.");

    auto [L, U] = this->lu();

    T det = 1;
    for (size_t i = 0; i < this->rows; ++i) {
        det *= U(i, i);
    }

    return det;
}

/* Matrix transformations */

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero for transposition.");
    }
    
    // If A is 1 x 1, then A^T = A
    if (this->rows == 1 && this->cols == 1) {
        return *this;
    }

    // If A is m x n, then A^T is n x m
    Matrix<T> transpose(this->cols, this->rows);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            // Flip A along its diagonal
            transpose(j, i) = this->data[i][j];
        }
    }

    return transpose;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the inverse.");
    
    T det = this->det();

    // If A is singular (i.e., det(A) = 0), then it does not have an inverse
    if (det == 0) {
        throw std::invalid_argument("Matrix must be non-singular to calculate the inverse.");
    }

    Matrix<T> adj(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            Matrix<T> submatrix(this->rows - 1, this->cols - 1);
    
            for (size_t k = 0; k < this->rows; ++k) {
                for (size_t l = 0; l < this->cols; ++l) {
                    if (k < i && l < j)
                        submatrix(k, l) = this->data[k][l];
                    else if (k < i && l > j)
                        submatrix(k, l - 1) = this->data[k][l];
                    else if (k > i && l < j)
                        submatrix(k - 1, l) = this->data[k][l];
                    else if (k > i && l > j)
                        submatrix(k - 1, l - 1) = this->data[k][l];
                }
            }

            adj(j, i) = ((i + j) % 2 == 0 ? 1 : -1) * submatrix.det();
        }
    }

    // The inverse of A is the adjugate of A divided by the determinant of A
    return adj * (1 / det);
}

template <typename T>
Matrix<T> Matrix<T>::adjoint() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the adjoint.");

    Matrix<T> adj(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            Matrix<T> submatrix(this->rows - 1, this->cols - 1);
    
            for (size_t k = 0; k < this->rows; ++k) {
                for (size_t l = 0; l < this->cols; ++l) {
                    if (k < i && l < j)
                        submatrix(k, l) = this->data[k][l];
                    else if (k < i && l > j)
                        submatrix(k, l - 1) = this->data[k][l];
                    else if (k > i && l < j)
                        submatrix(k - 1, l) = this->data[k][l];
                    else if (k > i && l > j)
                        submatrix(k - 1, l - 1) = this->data[k][l];
                }
            }

            adj(j, i) = ((i + j) % 2 == 0 ? 1 : -1) * submatrix.det();
        }
    }

    return adj;
}

template <typename T>
Matrix<T> Matrix<T>::conjugate() const {
    Matrix<T> conjugate(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            conjugate(i, j) = std::conj(this->data[i][j]);
        }
    }

    return conjugate;
}

template <typename T>
Matrix<T> Matrix<T>::exp() const {
    if (this->rows != this->cols)
        throw std::invalid_argument("Matrix must be square to calculate the exponential.");

    const int q = 6;
    const T norm = this->max_norm();
    const int s = std::max(0, static_cast<int>(std::ceil(std::log2(norm / q))));
    const T scale = std::pow(2, -s);

    Matrix<T> A = *this * scale;
    Matrix<T> P = Matrix<T>::identity(this->rows);
    Matrix<T> Q = Matrix<T>::identity(this->rows);

    const int num_terms = q;
    T c = 1;
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
    if (this->rows != this->cols) {
        throw std::invalid_argument("Matrix must be square to perform LU decomposition.");
    }
    
    // L is a lower triangular matrix with ones on the diagonal
    Matrix<T> L(this->rows, this->cols);

    // U is an upper triangular matrix
    Matrix<T> U(this->rows, this->cols);

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            if (i <= j) {
                U(i, j) = this->data[i][j];

                for (size_t k = 0; k < i; ++k) {
                    U(i, j) -= L(i, k) * U(k, j);
                }
    
                if (i == j) {
                    L(i, j) = 1;
                }
            } else {
                L(i, j) = this->data[i][j];

                for (size_t k = 0; k < j; ++k) {
                    L(i, j) -= L(i, k) * U(k, j);
                }

                L(i, j) /= U(j, j);
            }
        }
    }

    // Return the pair of matrices (L, U)
    return std::make_pair(L, U);
}

template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::qr() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero to perform QR decomposition.");
    }

    // Initialize Q as an identity matrix
    Matrix<T> Q = Matrix<T>::identity(this->rows);
    Matrix<T> R = *this;

    for (size_t i = 0; i < this->cols; ++i) {
        // Compute the Householder reflection
        T norm = 0;
        for (size_t j = i; j < this->rows; ++j) {
            norm += R(j, i) * R(j, i);
        }
        norm = std::sqrt(norm);

        T s = R(i, i) < 0 ? norm : -norm;
        T alpha = 1 / std::sqrt(2 * norm * (norm - R(i, i)));

        Vector<T> v(this->rows);
        v[i] = R(i, i) - s;
        for (size_t j = i + 1; j < this->rows; ++j) {
            v[j] = R(j, i);
        }

        // Update R using the Householder transformation
        R -= (v * v.transpose()) * (2 * alpha);

        // Update Q using the Householder transformation
        Q -= (Q * v) * (v.transpose() * Q) * (2 * alpha);
    }

    return std::make_pair(Q, R);
}

template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> Matrix<T>::svd() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero to perform SVD decomposition.");
    }

    // Compute the SVD using the QR algorithm
    Matrix<T> A = *this;
    Matrix<T> U = Matrix<T>::identity(this->rows);
    Matrix<T> V = Matrix<T>::identity(this->cols);

    const size_t max_iterations = 100;
    const T tolerance = 1e-6;

    for (size_t i = 0; i < max_iterations; ++i) {
        std::pair<Matrix<T>, Matrix<T>> qr = A.qr();
        Matrix<T> Q = qr.first;
        Matrix<T> R = qr.second;

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
    if (this->rows != this->cols) {
        throw std::invalid_argument("Matrix must be square to calculate its eigenvalues.");
    }
    
    if (this->det() == 0) {
        throw std::invalid_argument("Matrix must be non-singular to calculate its eigenvalues.");
    }
    
    // Use the power iteration method to find the eigenvalues
    const size_t max_iterations = 100;
    const T tolerance = 1e-6;
    
    std::vector<std::complex<T>> eigvals;
    Matrix<T> A = *this;
    
    for (size_t i = 0; i < this->rows; ++i) {
        Matrix<T> x(this->rows, 1);
        x(0, 0) = 1;
        
        for (size_t j = 0; j < max_iterations; ++j) {
            Matrix<T> y = A * x;
            x = y * (1 / y.max_norm());
            
            Matrix<T> z = A * x;
            T eigenvalue = (z.transpose() * x)(0, 0) / (x.transpose() * x)(0, 0);
            
            if ((z - x * eigenvalue).max_norm() < tolerance) {
                break;
            }
        }
        
        eigvals.push_back(std::complex<T>(A(i, i), 0));
        
        for (size_t j = i; j < this->rows; ++j) {
            A(j, i) = A(i, j) = 0;
        }

        A(i, i) = 1;
    }
    
    return eigvals;
}

template <typename T>
Matrix<T> Matrix<T>::eigenvectors() const {
    if (this->rows == 0 || this->cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero to calculate eigenvectors.");
    }

    if (this->rows != this->cols) {
        throw std::invalid_argument("Matrix must be square to calculate its eigenvectors.");
    }

    if (this->det() == 0) {
        throw std::invalid_argument("Matrix must be non-singular to calculate its eigenvectors.");
    }
    
    // Use the power iteration method to find the eigenvectors
    const size_t max_iterations = 100;
    const T tolerance = 1e-6;
    
    Matrix<T> eigvecs(this->rows, this->cols);
    Matrix<T> A = *this;
    
    for (size_t i = 0; i < this->rows; ++i) {
        Matrix<T> x(this->rows, 1);
        x(0, 0) = 1;
        
        for (size_t j = 0; j < max_iterations; ++j) {
            Matrix<T> y = A * x;
            x = y * (1 / y.max_norm());
        }
        
        for (size_t j = 0; j < this->rows; ++j) {
            eigvecs(j, i) = x(j, 0);
        }
        
        for (size_t j = i; j < this->rows; ++j) {
            A(j, i) = A(i, j) = 0;
        }

        A(i, i) = 1;
    }
    
    return eigvecs;
}

/* Helper functions */

template <typename T>
T Matrix<T>::max_norm() const {
    T norm = 0;

    for (size_t i = 0; i < this->rows; ++i) {
        T row_sum = 0;
        for (size_t j = 0; j < this->cols; ++j) {
            row_sum += std::abs(this->data[i][j]);
        }
        norm = std::max(norm, row_sum);
    }

    return norm;
}

} // namespace lincpp
