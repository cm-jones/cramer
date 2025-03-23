#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <algorithm>

#include "matrix.hpp"
#include "vector.hpp"

using namespace cramer;

class MatrixTest : public ::testing::Test {
protected:
    // Helper function to compare floating point numbers
    bool is_close(double first, double second, double tolerance = 1e-6) {
        return std::abs(first - second) < tolerance;
    }

    // Helper function to compare matrices
    template <typename T>
    bool matrices_are_close(const Matrix<T>& first, const Matrix<T>& second,
                           double tolerance = 1e-6) {
        if (first.get_rows() != second.get_rows() ||
            first.get_cols() != second.get_cols()) {
            return false;
        }
        for (size_t row = 0; row < first.get_rows(); ++row) {
            for (size_t col = 0; col < first.get_cols(); ++col) {
                if (!is_close(std::abs(first(row, col) - second(row, col)), 0.0, tolerance)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Helper function to compare vectors
    template <typename T>
    bool vectors_are_close(const Vector<T>& first, const Vector<T>& second,
                           double tolerance = 1e-6) {
        if (first.size() != second.size()) {
            return false;
        }
        for (size_t i = 0; i < first.size(); ++i) {
            if (!is_close(std::abs(first[i] - second[i]), 0.0, tolerance)) {
                return false;
            }
        }
        return true;
    }
};

// Basic construction and accessors
TEST_F(MatrixTest, DefaultConstructor) {
    Matrix<double> mat;
    EXPECT_EQ(mat.get_rows(), 0);
    EXPECT_EQ(mat.get_cols(), 0);
}

TEST_F(MatrixTest, ConstructorWithDimensions) {
    Matrix<double> mat(2, 3);
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 3);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            EXPECT_EQ(mat(row, col), 0.0);
        }
    }
}

TEST_F(MatrixTest, ConstructorWithDimensionsAndValue) {
    Matrix<double> mat(2, 3, 1.5);
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 3);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            EXPECT_DOUBLE_EQ(mat(row, col), 1.5);
        }
    }
}

TEST_F(MatrixTest, ConstructorWithVector) {
    std::vector<std::vector<double>> vec = {{1, 2}, {3, 4}};
    Matrix<double> mat(vec);
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 2);
    EXPECT_EQ(mat(0, 0), 1);
    EXPECT_EQ(mat(0, 1), 2);
    EXPECT_EQ(mat(1, 0), 3);
    EXPECT_EQ(mat(1, 1), 4);
}

TEST_F(MatrixTest, ConstructorWithInitializerList) {
    Matrix<double> mat(2, 2, {1, 2, 3, 4});
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 2);
    EXPECT_EQ(mat(0, 0), 1);
    EXPECT_EQ(mat(0, 1), 2);
    EXPECT_EQ(mat(1, 0), 3);
    EXPECT_EQ(mat(1, 1), 4);
}

// Special matrices
TEST_F(MatrixTest, IdentityMatrix) {
    Matrix<double> mat = Matrix<double>::identity(3);
    EXPECT_EQ(mat.get_rows(), 3);
    EXPECT_EQ(mat.get_cols(), 3);
    for (size_t row = 0; row < 3; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            EXPECT_EQ(mat(row, col), row == col ? 1.0 : 0.0);
        }
    }
}

TEST_F(MatrixTest, ZerosMatrix) {
    Matrix<double> mat = Matrix<double>::zeros(2, 3);
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 3);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            EXPECT_DOUBLE_EQ(mat(row, col), 0.0);
        }
    }
}

TEST_F(MatrixTest, OnesMatrix) {
    Matrix<double> mat = Matrix<double>::ones(2, 3);
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 3);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            EXPECT_EQ(mat(row, col), 1.0);
        }
    }
}

// Basic operations
TEST_F(MatrixTest, EqualityOperator) {
    Matrix<double> mat1(2, 2, 1.0);
    Matrix<double> mat2(2, 2, 1.0);
    Matrix<double> mat3(2, 2, 2.0);
    EXPECT_TRUE(mat1 == mat2);
    EXPECT_FALSE(mat1 == mat3);
}

TEST_F(MatrixTest, AdditionOperator) {
    Matrix<double> mat1(2, 2, 1.0);
    Matrix<double> mat2(2, 2, 2.0);
    Matrix<double> result = mat1 + mat2;
    EXPECT_EQ(result.get_rows(), 2);
    EXPECT_EQ(result.get_cols(), 2);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            EXPECT_DOUBLE_EQ(result(row, col), 3.0);
        }
    }
}

TEST_F(MatrixTest, AdditionAssignmentOperator) {
    Matrix<double> mat1(2, 2, 1.0);
    Matrix<double> mat2(2, 2, 2.0);
    mat1 += mat2;
    EXPECT_EQ(mat1.get_rows(), 2);
    EXPECT_EQ(mat1.get_cols(), 2);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            EXPECT_DOUBLE_EQ(mat1(row, col), 3.0);
        }
    }
}

TEST_F(MatrixTest, SubtractionOperator) {
    Matrix<double> mat1(2, 2, 3.0);
    Matrix<double> mat2(2, 2, 1.0);
    Matrix<double> result = mat1 - mat2;
    EXPECT_EQ(result.get_rows(), 2);
    EXPECT_EQ(result.get_cols(), 2);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            EXPECT_DOUBLE_EQ(result(row, col), 2.0);
        }
    }
}

TEST_F(MatrixTest, SubtractionAssignmentOperator) {
    Matrix<double> mat1(2, 2, 3.0);
    Matrix<double> mat2(2, 2, 1.0);
    mat1 -= mat2;
    EXPECT_EQ(mat1.get_rows(), 2);
    EXPECT_EQ(mat1.get_cols(), 2);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            EXPECT_DOUBLE_EQ(mat1(row, col), 2.0);
        }
    }
}

TEST_F(MatrixTest, ScalarMultiplicationOperator) {
    Matrix<double> mat(2, 2, 2.0);
    Matrix<double> result = mat * 3.0;
    EXPECT_EQ(result.get_rows(), 2);
    EXPECT_EQ(result.get_cols(), 2);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            EXPECT_DOUBLE_EQ(result(row, col), 6.0);
        }
    }
}

TEST_F(MatrixTest, ScalarMultiplicationAssignmentOperator) {
    Matrix<double> mat(2, 2, 2.0);
    mat *= 3.0;
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 2);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            EXPECT_DOUBLE_EQ(mat(row, col), 6.0);
        }
    }
}

TEST_F(MatrixTest, MatrixMultiplicationOperator) {
    Matrix<double> mat1(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix<double> mat2(3, 2, {7, 8, 9, 10, 11, 12});
    Matrix<double> result = mat1 * mat2;
    EXPECT_EQ(result.get_rows(), 2);
    EXPECT_EQ(result.get_cols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 154.0);
}

TEST_F(MatrixTest, MultiplyVector) {
    Matrix<double> mat(2, 3, {1, 2, 3, 4, 5, 6});
    Vector<double> vec({7, 8, 9});
    Vector<double> result = mat.multiply_vector(vec);
    EXPECT_EQ(result.size(), 2);
    EXPECT_DOUBLE_EQ(result[0], 50.0); // 1*7 + 2*8 + 3*9
    EXPECT_DOUBLE_EQ(result[1], 122.0); // 4*7 + 5*8 + 6*9
}

// Matrix properties
TEST_F(MatrixTest, IsSquare) {
    Matrix<double> mat1(2, 2);
    Matrix<double> mat2(2, 3);
    EXPECT_TRUE(mat1.is_square());
    EXPECT_FALSE(mat2.is_square());
}

TEST_F(MatrixTest, IsDiagonal) {
    Matrix<double> mat1(2, 2, {1, 0, 0, 2});
    Matrix<double> mat2(2, 2, {1, 1, 0, 2});
    EXPECT_TRUE(mat1.is_diagonal());
    EXPECT_FALSE(mat2.is_diagonal());
}

TEST_F(MatrixTest, IsSymmetric) {
    Matrix<double> mat1(2, 2, {1, 2, 2, 1});
    Matrix<double> mat2(2, 2, {1, 2, 3, 1});
    EXPECT_TRUE(mat1.is_symmetric());
    EXPECT_FALSE(mat2.is_symmetric());
}

TEST_F(MatrixTest, IsInvertible) {
    Matrix<double> mat1(2, 2, {1, 0, 0, 1});
    Matrix<double> mat2(2, 2, {1, 1, 1, 1});
    EXPECT_TRUE(mat1.is_invertible());
    EXPECT_FALSE(mat2.is_invertible());
}

TEST_F(MatrixTest, IsHermitian) {
    Matrix<std::complex<double>> mat1(
        2, 2,
        {std::complex<double>(1, 0), std::complex<double>(2, 1),
         std::complex<double>(2, -1), std::complex<double>(3, 0)});
    Matrix<std::complex<double>> mat2(
        2, 2,
        {std::complex<double>(1, 0), std::complex<double>(2, 1),
         std::complex<double>(2, 1), std::complex<double>(3, 0)});
    EXPECT_TRUE(mat1.is_hermitian());
    EXPECT_FALSE(mat2.is_hermitian());
}

TEST_F(MatrixTest, IsOrthogonal) {
    Matrix<double> mat1(2, 2, {0, 1, -1, 0});
    Matrix<double> mat2(2, 2, {1, 1, 1, 1});
    EXPECT_TRUE(mat1.is_orthogonal());
    EXPECT_FALSE(mat2.is_orthogonal());
}

TEST_F(MatrixTest, IsUnitary) {
    // A unitary matrix for complex numbers is similar to orthogonal for real numbers
    Matrix<std::complex<double>> mat(
        2, 2,
        {std::complex<double>(0, 1), std::complex<double>(1, 0),
         std::complex<double>(1, 0), std::complex<double>(0, -1)});
    EXPECT_TRUE(mat.is_unitary());
}

TEST_F(MatrixTest, Trace) {
    Matrix<double> mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_EQ(mat.trace(), 15);
}

TEST_F(MatrixTest, Determinant) {
    // Test 2x2 determinant
    Matrix<double> mat1(2, 2, {1, 2, 3, 4});
    EXPECT_DOUBLE_EQ(mat1.det(), -2.0); // 1*4 - 2*3

    // Test singular matrix (determinant should be 0)
    Matrix<double> mat2(2, 2, {1, 2, 2, 4});
    EXPECT_NEAR(mat2.det(), 0.0, 1e-9);
    
    // Test 3x3 matrix
    Matrix<double> mat3(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_NEAR(mat3.det(), 0, 1e-9);
}

TEST_F(MatrixTest, DeterminantViaLU) {
    Matrix<double> mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_NEAR(mat.det_via_lu(), 0, 1e-9);
}

TEST_F(MatrixTest, Transpose) {
    Matrix<double> mat(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix<double> result = mat.transpose();
    EXPECT_EQ(result.get_rows(), 3);
    EXPECT_EQ(result.get_cols(), 2);
    EXPECT_EQ(result(0, 0), 1);
    EXPECT_EQ(result(0, 1), 4);
    EXPECT_EQ(result(1, 0), 2);
    EXPECT_EQ(result(1, 1), 5);
    EXPECT_EQ(result(2, 0), 3);
    EXPECT_EQ(result(2, 1), 6);
}

TEST_F(MatrixTest, Inverse) {
    Matrix<double> mat(2, 2, {1, 2, 3, 4});
    Matrix<double> result = mat.inverse();
    EXPECT_NEAR(result(0, 0), -2, 1e-9);
    EXPECT_NEAR(result(0, 1), 1, 1e-9);
    EXPECT_NEAR(result(1, 0), 1.5, 1e-9);
    EXPECT_NEAR(result(1, 1), -0.5, 1e-9);
    
    // Check that A * A^-1 = I
    Matrix<double> identity = mat * result;
    EXPECT_TRUE(matrices_are_close(identity, Matrix<double>::identity(2), 1e-9));
}

TEST_F(MatrixTest, Adjoint) {
    Matrix<double> mat(2, 2, {1, 2, 3, 4});
    Matrix<double> result = mat.adjoint();
    EXPECT_EQ(result(0, 0), 4);
    EXPECT_EQ(result(0, 1), -2);
    EXPECT_EQ(result(1, 0), -3);
    EXPECT_EQ(result(1, 1), 1);
}

TEST_F(MatrixTest, Conjugate) {
    Matrix<std::complex<double>> mat(
        2, 2,
        {std::complex<double>(1, 1), std::complex<double>(2, 2),
         std::complex<double>(3, 3), std::complex<double>(4, 4)});
    Matrix<std::complex<double>> result = mat.conjugate();
    EXPECT_EQ(result(0, 0), std::complex<double>(1, -1));
    EXPECT_EQ(result(0, 1), std::complex<double>(2, -2));
    EXPECT_EQ(result(1, 0), std::complex<double>(3, -3));
    EXPECT_EQ(result(1, 1), std::complex<double>(4, -4));
}

TEST_F(MatrixTest, LU) {
    Matrix<double> mat(3, 3, {4, 3, 2, 1, 4, 3, 2, 1, 4});
    auto [lower, upper] = mat.lu();

    // Check lower triangular properties
    EXPECT_TRUE(lower.is_square());
    for (size_t row = 0; row < lower.get_rows(); ++row) {
        for (size_t col = 0; col < lower.get_cols(); ++col) {
            if (row == col) {
                EXPECT_NEAR(lower(row, col), 1, 1e-9);
            } else if (row < col) {
                EXPECT_NEAR(lower(row, col), 0, 1e-9);
            }
        }
    }

    // Check upper triangular properties
    EXPECT_TRUE(upper.is_square());
    for (size_t row = 0; row < upper.get_rows(); ++row) {
        for (size_t col = 0; col < upper.get_cols(); ++col) {
            if (row > col) {
                EXPECT_NEAR(upper(row, col), 0, 1e-9);
            }
        }
    }

    // Check that L * U = mat
    EXPECT_TRUE(matrices_are_close(lower * upper, mat));
}

TEST_F(MatrixTest, QR) {
    Matrix<double> mat(3, 3, {12, -51, 4, 6, 167, -68, -4, 24, -41});
    auto [orthogonal, upper] = mat.qr();

    // Check orthogonal properties (Q^T * Q = I)
    Matrix<double> QT_Q = orthogonal.transpose() * orthogonal;
    Matrix<double> I = Matrix<double>::identity(3);
    EXPECT_TRUE(matrices_are_close(QT_Q, I, 1e-6));

    // Check upper triangular properties
    for (size_t row = 0; row < upper.get_rows(); ++row) {
        for (size_t col = 0; col < upper.get_cols(); ++col) {
            if (row > col) {
                EXPECT_NEAR(upper(row, col), 0, 1e-6);
            }
        }
    }

    // Check that Q * R = mat
    Matrix<double> QR = orthogonal * upper;
    EXPECT_TRUE(matrices_are_close(QR, mat, 1e-6));
}

TEST_F(MatrixTest, SVD) {
    // Create a simple 2x2 diagonal matrix for testing SVD
    Matrix<double> mat(2, 2, {4, 0, 0, 9});
    auto [left, singular, right] = mat.svd();
    
    // Check dimensions
    EXPECT_EQ(left.get_rows(), 2);
    EXPECT_EQ(left.get_cols(), 2);
    EXPECT_EQ(singular.get_rows(), 2);
    EXPECT_EQ(singular.get_cols(), 2);
    EXPECT_EQ(right.get_rows(), 2);
    EXPECT_EQ(right.get_cols(), 2);
    
    // Check singular values
    EXPECT_NEAR(singular(0, 0), 9.0, 0.1);
    EXPECT_NEAR(singular(1, 1), 4.0, 0.1);
    EXPECT_NEAR(singular(0, 1), 0.0, 0.1);
    EXPECT_NEAR(singular(1, 0), 0.0, 0.1);
    
    // Check orthogonality
    Matrix<double> I2 = Matrix<double>::identity(2);
    Matrix<double> left_ortho = left * left.transpose();
    Matrix<double> right_ortho = right * right.transpose();
    
    EXPECT_TRUE(matrices_are_close(left_ortho, I2, 0.1));
    EXPECT_TRUE(matrices_are_close(right_ortho, I2, 0.1));
    
    // Check reconstruction
    Matrix<double> reconstructed = left * singular * right.transpose();
    EXPECT_TRUE(matrices_are_close(reconstructed, mat, 0.1));
}

TEST_F(MatrixTest, Eigenvalues) {
    Matrix<double> mat(2, 2, {3, 0, 0, 2});
    auto eigenvals = mat.eigenvalues();
    
    // Check we have the right number of eigenvalues
    EXPECT_EQ(eigenvals.size(), 2);
    
    // Sort eigenvalues for consistent testing
    std::vector<double> eigenval_reals;
    for (const auto& val : eigenvals) {
        eigenval_reals.push_back(std::real(val));
    }
    std::sort(eigenval_reals.begin(), eigenval_reals.end(), std::greater<double>());
    
    // For a diagonal matrix [3 0; 0 2], eigenvalues should be 3 and 2
    EXPECT_NEAR(eigenval_reals[0], 3.0, 0.01);
    EXPECT_NEAR(eigenval_reals[1], 2.0, 0.01);
}

TEST_F(MatrixTest, Eigenvectors) {
    Matrix<double> mat(2, 2, {3, 0, 0, 2});
    Matrix<double> eigenvecs = mat.eigenvectors();
    
    EXPECT_EQ(eigenvecs.get_rows(), 2);
    EXPECT_EQ(eigenvecs.get_cols(), 2);
    
    // Check if eigenvectors are normalized (unit vectors)
    for (size_t col = 0; col < 2; ++col) {
        double norm_squared = 0.0;
        for (size_t row = 0; row < 2; ++row) {
            norm_squared += eigenvecs(row, col) * eigenvecs(row, col);
        }
        EXPECT_NEAR(norm_squared, 1.0, 0.01);
    }
    
    // For diagonal matrix [3 0; 0 2], eigenvectors should be basis vectors
    
    // Since eigenvectors can vary in sign and order, we need to be flexible in testing
    // Check that each eigenvector is aligned with one of the coordinate axes
    bool found_x_axis = false;
    bool found_y_axis = false;
    
    for (size_t col = 0; col < 2; ++col) {
        if (std::abs(eigenvecs(0, col)) > 0.99 && std::abs(eigenvecs(1, col)) < 0.01) {
            // This is close to [±1, 0]
            found_x_axis = true;
            
            // Verify Av = λv
            Vector<double> v(2);
            v[0] = eigenvecs(0, col);
            v[1] = eigenvecs(1, col);
            Vector<double> Av = mat.multiply_vector(v);
            EXPECT_NEAR(Av[0], 3.0 * v[0], 0.01);  // λ₁ = 3
        }
        else if (std::abs(eigenvecs(1, col)) > 0.99 && std::abs(eigenvecs(0, col)) < 0.01) {
            // This is close to [0, ±1]
            found_y_axis = true;
            
            // Verify Av = λv
            Vector<double> v(2);
            v[0] = eigenvecs(0, col);
            v[1] = eigenvecs(1, col);
            Vector<double> Av = mat.multiply_vector(v);
            EXPECT_NEAR(Av[1], 2.0 * v[1], 0.01);  // λ₂ = 2
        }
    }
    
    EXPECT_TRUE(found_x_axis);
    EXPECT_TRUE(found_y_axis);
}

TEST_F(MatrixTest, Solve) {
    Matrix<double> mat(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Vector<double> vec({1, -2, 0});
    Vector<double> solution = mat.solve(vec);
    
    // Check Ax = b
    Vector<double> result = mat.multiply_vector(solution);
    EXPECT_TRUE(vectors_are_close(result, vec));
}

TEST_F(MatrixTest, ComplexMatrixOperations) {
    using Complex = std::complex<double>;
    
    Matrix<Complex> mat(2, 2, {
        Complex(1, 1), Complex(2, 2),
        Complex(3, 3), Complex(4, 4)
    });
    
    // Test conjugate
    Matrix<Complex> conj = mat.conjugate();
    EXPECT_EQ(conj(0, 0), Complex(1, -1));
    EXPECT_EQ(conj(0, 1), Complex(2, -2));
    EXPECT_EQ(conj(1, 0), Complex(3, -3));
    EXPECT_EQ(conj(1, 1), Complex(4, -4));
    
    // Test is_hermitian
    Matrix<Complex> herm(2, 2, {
        Complex(1, 0), Complex(2, 1),
        Complex(2, -1), Complex(3, 0)
    });
    EXPECT_TRUE(herm.is_hermitian());
    EXPECT_FALSE(mat.is_hermitian());
}

// Test for matrix exponential, power, sqrt, and log
TEST_F(MatrixTest, MatrixExponential) {
    Matrix<double> mat(2, 2, {0, 1, 0, 0});  // Nilpotent matrix
    Matrix<double> exp_mat = mat.exp();
    
    // For this nilpotent matrix, exp(A) = I + A
    Matrix<double> expected(2, 2, {1, 1, 0, 1});
    EXPECT_TRUE(matrices_are_close(exp_mat, expected, 1e-6));
}

TEST_F(MatrixTest, MatrixPower) {
    Matrix<double> mat(2, 2, {1, 1, 0, 1});
    
    // Test power 2
    Matrix<double> pow2 = mat.pow(2);
    Matrix<double> expected2(2, 2, {1, 2, 0, 1});
    EXPECT_TRUE(matrices_are_close(pow2, expected2, 1e-6));
    
    // Test power 3
    Matrix<double> pow3 = mat.pow(3);
    Matrix<double> expected3(2, 2, {1, 3, 0, 1});
    EXPECT_TRUE(matrices_are_close(pow3, expected3, 1e-6));
    
    // Test power 0 (should be identity)
    Matrix<double> pow0 = mat.pow(0);
    EXPECT_TRUE(matrices_are_close(pow0, Matrix<double>::identity(2), 1e-6));
    
    // Test negative power
    Matrix<double> mat_inv(2, 2, {1, -1, 0, 1});
    Matrix<double> pow_neg1 = mat.pow(-1);
    EXPECT_TRUE(matrices_are_close(pow_neg1, mat_inv, 1e-6));
}

TEST_F(MatrixTest, MatrixSqrt) {
    // Test with identity matrix (sqrt(I) = I)
    Matrix<double> identity = Matrix<double>::identity(2);
    Matrix<double> sqrt_identity = identity.sqrt();
    EXPECT_TRUE(matrices_are_close(sqrt_identity, identity, 1e-6));
    
    // Test with diagonal matrix
    Matrix<double> diag(2, 2, {4, 0, 0, 9});
    Matrix<double> sqrt_diag = diag.sqrt();
    Matrix<double> expected_sqrt(2, 2, {2, 0, 0, 3});
    EXPECT_TRUE(matrices_are_close(sqrt_diag, expected_sqrt, 1e-6));
}

TEST_F(MatrixTest, MatrixLog) {
    // Test with identity matrix (log(I) = 0)
    Matrix<double> identity = Matrix<double>::identity(2);
    Matrix<double> log_identity = identity.log();
    Matrix<double> zero = Matrix<double>::zeros(2, 2);
    EXPECT_TRUE(matrices_are_close(log_identity, zero, 1e-6));
    
    // Test with diagonal matrix
    Matrix<double> diag(2, 2, {std::exp(1.0), 0, 0, std::exp(2.0)});
    Matrix<double> log_diag = diag.log();
    Matrix<double> expected_log(2, 2, {1.0, 0, 0, 2.0});
    EXPECT_TRUE(matrices_are_close(log_diag, expected_log, 1e-6));
}

TEST_F(MatrixTest, OuterProduct) {
    Vector<double> u({1, 2, 3});
    Vector<double> v({4, 5});
    
    Matrix<double> outer = Matrix<double>::outer(u, v);
    
    EXPECT_EQ(outer.get_rows(), 3);
    EXPECT_EQ(outer.get_cols(), 2);
    
    EXPECT_DOUBLE_EQ(outer(0, 0), 4.0);  // 1 * 4
    EXPECT_DOUBLE_EQ(outer(0, 1), 5.0);  // 1 * 5
    EXPECT_DOUBLE_EQ(outer(1, 0), 8.0);  // 2 * 4
    EXPECT_DOUBLE_EQ(outer(1, 1), 10.0); // 2 * 5
    EXPECT_DOUBLE_EQ(outer(2, 0), 12.0); // 3 * 4
    EXPECT_DOUBLE_EQ(outer(2, 1), 15.0); // 3 * 5
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
