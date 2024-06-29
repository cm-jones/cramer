#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "matrix.hpp"
#include "vector.hpp"

using namespace cramer;

class MatrixTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Common setup for each test
    }

    // Helper function to compare floating point numbers
    bool isClose(double first, double second, double tolerance = 1e-9) {
        return std::abs(first - second) < tolerance;
    }

    // Helper function to compare matrices
    template <typename T>
    bool matricesAreClose(const Matrix<T>& first, const Matrix<T>& second,
                          double tolerance = 1e-9) {
        if (first.get_rows() != second.get_rows() ||
            first.get_cols() != second.get_cols()) {
            return false;
        }
        for (size_t row = 0; row < first.get_rows(); ++row) {
            for (size_t col = 0; col < first.get_cols(); ++col) {
                if (!isClose(first(row, col), second(row, col), tolerance)) {
                    return false;
                }
            }
        }
        return true;
    }
};

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
            EXPECT_EQ(mat(row, col), 1.5);
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
            EXPECT_EQ(mat(row, col), 0.0);
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

TEST_F(MatrixTest, RandomMatrix) {
    Matrix<double> mat = Matrix<double>::random(2, 3);
    EXPECT_EQ(mat.get_rows(), 2);
    EXPECT_EQ(mat.get_cols(), 3);
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
            EXPECT_GE(mat(row, col), 0.0);
            EXPECT_LT(mat(row, col), 1.0);
        }
    }
}

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
            EXPECT_EQ(result(row, col), 3.0);
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
            EXPECT_EQ(mat1(row, col), 3.0);
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
            EXPECT_EQ(result(row, col), 2.0);
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
            EXPECT_EQ(mat1(row, col), 2.0);
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
            EXPECT_EQ(result(row, col), 6.0);
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
            EXPECT_EQ(mat(row, col), 6.0);
        }
    }
}

TEST_F(MatrixTest, MatrixMultiplicationOperator) {
    Matrix<double> mat1(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix<double> mat2(3, 2, {7, 8, 9, 10, 11, 12});
    Matrix<double> result = mat1 * mat2;
    EXPECT_EQ(result.get_rows(), 2);
    EXPECT_EQ(result.get_cols(), 2);
    EXPECT_EQ(result(0, 0), 58);
    EXPECT_EQ(result(0, 1), 64);
    EXPECT_EQ(result(1, 0), 139);
    EXPECT_EQ(result(1, 1), 154);
}

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

TEST_F(MatrixTest, Trace) {
    Matrix<double> mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_EQ(mat.trace(), 15);
}

TEST_F(MatrixTest, Determinant) {
    Matrix<double> mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_NEAR(mat.det(), 0, 1e-9);
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

TEST_F(MatrixTest, Exp) {
    Matrix<double> mat(2, 2, {0, 1, -1, 0});
    Matrix<double> result = mat.exp();
    EXPECT_NEAR(result(0, 0), std::cos(1), 1e-9);
    EXPECT_NEAR(result(0, 1), std::sin(1), 1e-9);
    EXPECT_NEAR(result(1, 0), -std::sin(1), 1e-9);
    EXPECT_NEAR(result(1, 1), std::cos(1), 1e-9);
}

TEST_F(MatrixTest, Pow) {
    Matrix<double> mat(2, 2, {1, 1, 1, 0});
    Matrix<double> result = mat.pow(3);
    EXPECT_EQ(result(0, 0), 3);
    EXPECT_EQ(result(0, 1), 2);
    EXPECT_EQ(result(1, 0), 2);
    EXPECT_EQ(result(1, 1), 1);
}

TEST_F(MatrixTest, Sqrt) {
    Matrix<double> mat(2, 2, {4, 0, 0, 9});
    Matrix<double> result = mat.sqrt();
    EXPECT_NEAR(result(0, 0), 2, 1e-9);
    EXPECT_NEAR(result(0, 1), 0, 1e-9);
    EXPECT_NEAR(result(1, 0), 0, 1e-9);
    EXPECT_NEAR(result(1, 1), 3, 1e-9);
}

TEST_F(MatrixTest, Log) {
    Matrix<double> mat(2, 2, {std::exp(1), 0, 0, std::exp(2)});
    Matrix<double> result = mat.log();
    EXPECT_NEAR(result(0, 0), 1, 1e-9);
    EXPECT_NEAR(result(0, 1), 0, 1e-9);
    EXPECT_NEAR(result(1, 0), 0, 1e-9);
    EXPECT_NEAR(result(1, 1), 2, 1e-9);
}

TEST_F(MatrixTest, LU) {
    Matrix<double> mat(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
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
    EXPECT_TRUE(matricesAreClose(lower * upper, mat));
}

TEST_F(MatrixTest, QR) {
    Matrix<double> mat(3, 3, {12, -51, 4, 6, 167, -68, -4, 24, -41});
    auto [orthogonal, upper] = mat.qr();

    // Check orthogonal properties
    EXPECT_TRUE(orthogonal.is_square());
    EXPECT_TRUE(matricesAreClose(orthogonal * orthogonal.transpose(),
                                 Matrix<double>::identity(3)));

    // Check upper triangular properties
    EXPECT_TRUE(upper.is_square());
    for (size_t row = 0; row < upper.get_rows(); ++row) {
        for (size_t col = 0; col < upper.get_cols(); ++col) {
            if (row > col) {
                EXPECT_NEAR(upper(row, col), 0, 1e-9);
            }
        }
    }

    // Check that Q * R = mat
    EXPECT_TRUE(matricesAreClose(orthogonal * upper, mat));
}

TEST_F(MatrixTest, SVD) {
    Matrix<double> mat(3, 2, {1, 2, 3, 4, 5, 6});
    auto [left, singular, right] = mat.svd();

    // Check left singular vector properties
    EXPECT_EQ(left.get_rows(), mat.get_rows());
    EXPECT_EQ(left.get_cols(), mat.get_rows());
    EXPECT_TRUE(matricesAreClose(left * left.transpose(),
                                 Matrix<double>::identity(mat.get_rows())));

    // Check singular value properties
    EXPECT_EQ(singular.get_rows(), mat.get_rows());
    EXPECT_EQ(singular.get_cols(), mat.get_cols());
    for (size_t row = 0; row < singular.get_rows(); ++row) {
        for (size_t col = 0; col < singular.get_cols(); ++col) {
            if (row != col) {
                EXPECT_NEAR(singular(row, col), 0, 1e-9);
            } else {
                EXPECT_GE(singular(row, col), 0);
            }
        }
    }

    // Check right singular vector properties
    EXPECT_EQ(right.get_rows(), mat.get_cols());
    EXPECT_EQ(right.get_cols(), mat.get_cols());
    EXPECT_TRUE(matricesAreClose(right * right.transpose(),
                                 Matrix<double>::identity(mat.get_cols())));

    // Check that U * S * V^T = mat
    EXPECT_TRUE(matricesAreClose(left * singular * right.transpose(), mat));
}

TEST_F(MatrixTest, Eigenvalues) {
    Matrix<double> mat(2, 2, {1, 2, 2, 1});
    auto eigenvals = mat.eigenvalues();
    EXPECT_EQ(eigenvals.size(), 2);
    EXPECT_NEAR(std::abs(eigenvals[0]), 3, 1e-9);
    EXPECT_NEAR(std::abs(eigenvals[1]), 1, 1e-9);
}

TEST_F(MatrixTest, Eigenvectors) {
    Matrix<double> mat(2, 2, {1, 2, 2, 1});
    Matrix<double> eigenvecs = mat.eigenvectors();
    EXPECT_EQ(eigenvecs.get_rows(), 2);
    EXPECT_EQ(eigenvecs.get_cols(), 2);

    // Check that (A - lambda * I) * v = 0 for each eigenpair
    auto eigenvals = mat.eigenvalues();
    for (size_t idx = 0; idx < eigenvals.size(); ++idx) {
        Matrix<double> vec(2, 1);
        vec(0, 0) = eigenvecs(0, idx);
        vec(1, 0) = eigenvecs(1, idx);
        Matrix<double> result =
            (mat - Matrix<double>::identity(2) * std::real(eigenvals[idx])) *
            vec;
        EXPECT_NEAR(result(0, 0), 0, 1e-9);
        EXPECT_NEAR(result(1, 0), 0, 1e-9);
    }
}

TEST_F(MatrixTest, Rank) {
    Matrix<double> mat1(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    EXPECT_EQ(mat1.rank(), 3);

    Matrix<double> mat2(3, 3, {1, 2, 3, 2, 4, 6, 3, 6, 9});
    EXPECT_EQ(mat2.rank(), 1);
}

TEST_F(MatrixTest, Solve) {
    Matrix<double> mat(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Vector<double> vec({1, -2, 0});
    Vector<double> solution = mat.solve(vec);

    // Check Ax = b
    Vector<double> result = mat * solution;
    EXPECT_NEAR(result[0], vec[0], 1e-9);
    EXPECT_NEAR(result[1], vec[1], 1e-9);
    EXPECT_NEAR(result[2], vec[2], 1e-9);
}

// Main function to run the tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
