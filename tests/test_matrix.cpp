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

#include "../include/matrix.h"
#include "../include/vector.h"
#include "gtest/gtest.h"

using namespace lincpp;

TEST(MatrixTest, ConstructorsAndGetters) {
    Matrix<int> m1;
    EXPECT_EQ(m1.get_rows(), 0);
    EXPECT_EQ(m1.get_cols(), 0);

    Matrix<int> m2(3, 4);
    EXPECT_EQ(m2.get_rows(), 3);
    EXPECT_EQ(m2.get_cols(), 4);

    Matrix<int> m3(2, 3, 5);
    EXPECT_EQ(m3.get_rows(), 2);
    EXPECT_EQ(m3.get_cols(), 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(m3(i, j), 5);
        }
    }

    std::vector<std::vector<int>> values = {{1, 2}, {3, 4}};
    Matrix<int> m4(values);
    EXPECT_EQ(m4.get_rows(), 2);
    EXPECT_EQ(m4.get_cols(), 2);
    EXPECT_EQ(m4(0, 0), 1);
    EXPECT_EQ(m4(0, 1), 2);
    EXPECT_EQ(m4(1, 0), 3);
    EXPECT_EQ(m4(1, 1), 4);
}

TEST(MatrixTest, SpecialMatrices) {
    Matrix<double> I = Matrix<double>::identity(3);
    EXPECT_EQ(I.get_rows(), 3);
    EXPECT_EQ(I.get_cols(), 3);
    EXPECT_EQ(I(0, 0), 1);
    EXPECT_EQ(I(0, 1), 0);
    EXPECT_EQ(I(0, 2), 0);
    EXPECT_EQ(I(1, 0), 0);
    EXPECT_EQ(I(1, 1), 1);
    EXPECT_EQ(I(1, 2), 0);
    EXPECT_EQ(I(2, 0), 0);
    EXPECT_EQ(I(2, 1), 0);
    EXPECT_EQ(I(2, 2), 1);

    Matrix<double> Z = Matrix<double>::zeros(2, 3);
    EXPECT_EQ(Z.get_rows(), 2);
    EXPECT_EQ(Z.get_cols(), 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(Z(i, j), 0);
        }
    }

    Matrix<double> O = Matrix<double>::ones(3, 2);
    EXPECT_EQ(O.get_rows(), 3);
    EXPECT_EQ(O.get_cols(), 2);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(O(i, j), 1);
        }
    }
}

TEST(MatrixTest, Operators) {
    Matrix<int> m1(2, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;

    Matrix<int> m2(2, 3);
    m2(0, 0) = 7;
    m2(0, 1) = 8;
    m2(0, 2) = 9;
    m2(1, 0) = 10;
    m2(1, 1) = 11;
    m2(1, 2) = 12;

    Matrix<int> m3 = m1 + m2;
    EXPECT_EQ(m3(0, 0), 8);
    EXPECT_EQ(m3(0, 1), 10);
    EXPECT_EQ(m3(0, 2), 12);
    EXPECT_EQ(m3(1, 0), 14);
    EXPECT_EQ(m3(1, 1), 16);
    EXPECT_EQ(m3(1, 2), 18);

    m3 += m2;
    EXPECT_EQ(m3(0, 0), 15);
    EXPECT_EQ(m3(0, 1), 18);
    EXPECT_EQ(m3(0, 2), 21);
    EXPECT_EQ(m3(1, 0), 24);
    EXPECT_EQ(m3(1, 1), 27);
    EXPECT_EQ(m3(1, 2), 30);

    Matrix<int> m4 = m3 - m2;
    EXPECT_EQ(m4(0, 0), 8);
    EXPECT_EQ(m4(0, 1), 10);
    EXPECT_EQ(m4(0, 2), 12);
    EXPECT_EQ(m4(1, 0), 14);
    EXPECT_EQ(m4(1, 1), 16);
    EXPECT_EQ(m4(1, 2), 18);

    m4 -= m1;
    EXPECT_EQ(m4(0, 0), 7);
    EXPECT_EQ(m4(0, 1), 8);
    EXPECT_EQ(m4(0, 2), 9);
    EXPECT_EQ(m4(1, 0), 10);
    EXPECT_EQ(m4(1, 1), 11);
    EXPECT_EQ(m4(1, 2), 12);

    Matrix<int> m5 = m1 * 2;
    EXPECT_EQ(m5(0, 0), 2);
    EXPECT_EQ(m5(0, 1), 4);
    EXPECT_EQ(m5(0, 2), 6);
    EXPECT_EQ(m5(1, 0), 8);
    EXPECT_EQ(m5(1, 1), 10);
    EXPECT_EQ(m5(1, 2), 12);

    m5 *= 3;
    EXPECT_EQ(m5(0, 0), 6);
    EXPECT_EQ(m5(0, 1), 12);
    EXPECT_EQ(m5(0, 2), 18);
    EXPECT_EQ(m5(1, 0), 24);
    EXPECT_EQ(m5(1, 1), 30);
    EXPECT_EQ(m5(1, 2), 36);

    Matrix<int> m6(3, 2);
    m6(0, 0) = 1;
    m6(0, 1) = 2;
    m6(1, 0) = 3;
    m6(1, 1) = 4;
    m6(2, 0) = 5;
    m6(2, 1) = 6;

    Matrix<int> m7 = m1 * m6;
    EXPECT_EQ(m7.get_rows(), 2);
    EXPECT_EQ(m7.get_cols(), 2);
    EXPECT_EQ(m7(0, 0), 22);
    EXPECT_EQ(m7(0, 1), 28);
    EXPECT_EQ(m7(1, 0), 49);
    EXPECT_EQ(m7(1, 1), 64);
}

TEST(MatrixTest, Properties) {
    Matrix<int> m1(2, 2);
    m1(0, 0) = 1;
    m1(0, 1) = 0;
    m1(1, 0) = 0;
    m1(1, 1) = 1;
    EXPECT_TRUE(m1.is_square());
    EXPECT_TRUE(m1.is_diagonal());
    EXPECT_TRUE(m1.is_symmetric());
    EXPECT_TRUE(m1.is_invertible());
    EXPECT_TRUE(m1.is_hermitian());
    EXPECT_TRUE(m1.is_orthogonal());
    EXPECT_EQ(m1.trace(), 2);
    EXPECT_EQ(m1.det(), 1);
    EXPECT_EQ(m1.det_via_lu(), 1);

    Matrix<double> m2(2, 2);
    m2(0, 0) = 1;
    m2(0, 1) = 2;
    m2(1, 0) = 2;
    m2(1, 1) = 1;
    EXPECT_TRUE(m2.is_square());
    EXPECT_FALSE(m2.is_diagonal());
    EXPECT_TRUE(m2.is_symmetric());
    EXPECT_TRUE(m2.is_invertible());
    EXPECT_TRUE(m2.is_hermitian());
    EXPECT_FALSE(m2.is_orthogonal());
    EXPECT_EQ(m2.trace(), 2);
    EXPECT_EQ(m2.det(), -3);
    EXPECT_EQ(m2.det_via_lu(), -3);

    Matrix<std::complex<double>> m3(2, 2);
    m3(0, 0) = std::complex<double>(1, 0);
    m3(0, 1) = std::complex<double>(0, 1);
    m3(1, 0) = std::complex<double>(0, -1);
    m3(1, 1) = std::complex<double>(1, 0);
    EXPECT_TRUE(m3.is_square());
    EXPECT_FALSE(m3.is_diagonal());
    EXPECT_FALSE(m3.is_symmetric());
    EXPECT_TRUE(m3.is_invertible());
    EXPECT_TRUE(m3.is_hermitian());
    EXPECT_TRUE(m3.is_orthogonal());
    EXPECT_EQ(m3.trace(), std::complex<double>(2, 0));
    EXPECT_EQ(m3.det(), std::complex<double>(1, 0));
    EXPECT_EQ(m3.det_via_lu(), std::complex<double>(1, 0));
}

TEST(MatrixTest, MatrixFunctions) {
    Matrix<double> m1(2, 2);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(1, 0) = 3;
    m1(1, 1) = 4;

    Matrix<double> m2 = m1.exp();
    EXPECT_NEAR(m2(0, 0), 14.7785, 1e-4);
    EXPECT_NEAR(m2(0, 1), 11.0232, 1e-4);
    EXPECT_NEAR(m2(1, 0), 16.2032, 1e-4);
    EXPECT_NEAR(m2(1, 1), 12.0965, 1e-4);

    Matrix<double> m3 = m1.pow(3);
    EXPECT_EQ(m3(0, 0), 37);
    EXPECT_EQ(m3(0, 1), 54);
    EXPECT_EQ(m3(1, 0), 81);
    EXPECT_EQ(m3(1, 1), 118);

    Matrix<double> m4(2, 2);
    m4(0, 0) = 2;
    m4(0, 1) = 1;
    m4(1, 0) = std::log(2);
    m4(1, 1) = 0;

    Matrix<double> m5 = m4.log();
    EXPECT_NEAR(m5(0, 0), std::log(2), 1e-6);
    EXPECT_NEAR(m5(0, 1), 1, 1e-6);
    EXPECT_NEAR(m5(1, 0), std::log(2), 1e-6);
    EXPECT_NEAR(m5(1, 1), 0, 1e-6);
}

TEST(MatrixTest, Decompositions) {
    Matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
    m1(2, 1) = 8;
    m1(2, 2) = 9;

    auto [L, U] = m1.lu();
    EXPECT_NEAR(L(0, 0), 1, 1e-6);
    EXPECT_NEAR(L(0, 1), 0, 1e-6);
    EXPECT_NEAR(L(0, 2), 0, 1e-6);
    EXPECT_NEAR(L(1, 0), 4, 1e-6);
    EXPECT_NEAR(L(1, 1), 1, 1e-6);
    EXPECT_NEAR(L(1, 2), 0, 1e-6);
    EXPECT_NEAR(L(2, 0), 7, 1e-6);
    EXPECT_NEAR(L(2, 1), 2.66667, 1e-6);
    EXPECT_NEAR(L(2, 2), 1, 1e-6);

    EXPECT_NEAR(U(0, 0), 1, 1e-6);
    EXPECT_NEAR(U(0, 1), 2, 1e-6);
    EXPECT_NEAR(U(0, 2), 3, 1e-6);
    EXPECT_NEAR(U(1, 0), 0, 1e-6);
    EXPECT_NEAR(U(1, 1), -1, 1e-6);
    EXPECT_NEAR(U(1, 2), -3, 1e-6);
    EXPECT_NEAR(U(2, 0), 0, 1e-6);
    EXPECT_NEAR(U(2, 1), 0, 1e-6);
    EXPECT_NEAR(U(2, 2), 6, 1e-6);

    auto [Q, R] = m1.qr();
    EXPECT_NEAR(Q(0, 0), 0.22942, 1e-5);
    EXPECT_NEAR(Q(0, 1), 0.52924, 1e-5);
    EXPECT_NEAR(Q(0, 2), 0.81766, 1e-5);
    EXPECT_NEAR(Q(1, 0), 0.57356, 1e-5);
    EXPECT_NEAR(Q(1, 1), 0.18405, 1e-5);
    EXPECT_NEAR(Q(1, 2), -0.79721, 1e-5);
    EXPECT_NEAR(Q(2, 0), 0.78770, 1e-5);
    EXPECT_NEAR(Q(2, 1), -0.57143, 1e-5);
    EXPECT_NEAR(Q(2, 2), 0.20412, 1e-5);

    EXPECT_NEAR(R(0, 0), -12.92278, 1e-5);
    EXPECT_NEAR(R(0, 1), -6.72201, 1e-5);
    EXPECT_NEAR(R(0, 2), -2.91621, 1e-5);
    EXPECT_NEAR(R(1, 0), 0, 1e-6);
    EXPECT_NEAR(R(1, 1), -6.72733, 1e-5);
    EXPECT_NEAR(R(1, 2), -5.33141, 1e-5);
    EXPECT_NEAR(R(2, 0), 0, 1e-6);
    EXPECT_NEAR(R(2, 1), 0, 1e-6);
    EXPECT_NEAR(R(2, 2), 1.97986, 1e-5);

    auto [U, S, V] = m1.svd();
    EXPECT_NEAR(U(0, 0), 0.22942, 1e-5);
    EXPECT_NEAR(U(0, 1), 0.52924, 1e-5);
    EXPECT_NEAR(U(0, 2), 0.81766, 1e-5);
    EXPECT_NEAR(U(1, 0), 0.57356, 1e-5);
    EXPECT_NEAR(U(1, 1), 0.18405, 1e-5);
    EXPECT_NEAR(U(1, 2), -0.79721, 1e-5);
    EXPECT_NEAR(U(2, 0), 0.78770, 1e-5);
    EXPECT_NEAR(U(2, 1), -0.57143, 1e-5);
    EXPECT_NEAR(U(2, 2), 0.20412, 1e-5);

    EXPECT_NEAR(S(0, 0), 16.11625, 1e-5);
    EXPECT_NEAR(S(0, 1), 0, 1e-6);
    EXPECT_NEAR(S(0, 2), 0, 1e-6);
    EXPECT_NEAR(S(1, 0), 0, 1e-6);
    EXPECT_NEAR(S(1, 1), 5.70297, 1e-5);
    EXPECT_NEAR(S(1, 2), 0, 1e-6);
    EXPECT_NEAR(S(2, 0), 0, 1e-6);
    EXPECT_NEAR(S(2, 1), 0, 1e-6);
    EXPECT_NEAR(S(2, 2), 1.18078, 1e-5);

    EXPECT_NEAR(V(0, 0), 0.42163, 1e-5);
    EXPECT_NEAR(V(0, 1), 0.67749, 1e-5);
    EXPECT_NEAR(V(0, 2), 0.60262, 1e-5);
    EXPECT_NEAR(V(1, 0), -0.72962, 1e-5);
    EXPECT_NEAR(V(1, 1), 0.25066, 1e-5);
    EXPECT_NEAR(V(1, 2), 0.63595, 1e-5);
    EXPECT_NEAR(V(2, 0), 0.53852, 1e-5);
    EXPECT_NEAR(V(2, 1), -0.69037, 1e-5);
    EXPECT_NEAR(V(2, 2), 0.48091, 1e-5);
}

TEST(MatrixTest, EigenvaluesAndEigenvectors) {
    Matrix<double> m1(2, 2);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(1, 0) = 2;
    m1(1, 1) = 3;

    auto eigvals = m1.eigenvalues();
    EXPECT_NEAR(std::real(eigvals[0]), -1, 1e-6);
    EXPECT_NEAR(std::imag(eigvals[0]), 0, 1e-6);
    EXPECT_NEAR(std::real(eigvals[1]), 5, 1e-6);
    EXPECT_NEAR(std::imag(eigvals[1]), 0, 1e-6);

    Matrix<double> eigvecs = m1.eigenvectors();
    EXPECT_NEAR(eigvecs(0, 0), -0.89442, 1e-5);
    EXPECT_NEAR(eigvecs(0, 1), 0.44721, 1e-5);
    EXPECT_NEAR(eigvecs(1, 0), 0.44721, 1e-5);
    EXPECT_NEAR(eigvecs(1, 1), 0.89442, 1e-5);

    Matrix<std::complex<double>> m2(2, 2);
    m2(0, 0) = std::complex<double>(1, 0);
    m2(0, 1) = std::complex<double>(1, 1);
    m2(1, 0) = std::complex<double>(0, 0);
    m2(1, 1) = std::complex<double>(2, 0);

    auto eigvals2 = m2.eigenvalues();
    EXPECT_NEAR(std::real(eigvals2[0]), 2, 1e-6);
    EXPECT_NEAR(std::imag(eigvals2[0]), 1, 1e-6);
    EXPECT_NEAR(std::real(eigvals2[1]), 1, 1e-6);
    EXPECT_NEAR(std::imag(eigvals2[1]), -1, 1e-6);

    Matrix<std::complex<double>> eigvecs2 = m2.eigenvectors();
    EXPECT_NEAR(std::real(eigvecs2(0, 0)), 0.89442, 1e-5);
    EXPECT_NEAR(std::imag(eigvecs2(0, 0)), 0.44721, 1e-5);
    EXPECT_NEAR(std::real(eigvecs2(0, 1)), 0.44721, 1e-5);
    EXPECT_NEAR(std::imag(eigvecs2(0, 1)), -0.89442, 1e-5);
    EXPECT_NEAR(std::real(eigvecs2(1, 0)), 0.44721, 1e-5);
    EXPECT_NEAR(std::imag(eigvecs2(1, 0)), 0.89442, 1e-5);
    EXPECT_NEAR(std::real(eigvecs2(1, 1)), 0.89442, 1e-5);
    EXPECT_NEAR(std::imag(eigvecs2(1, 1)), 0.44721, 1e-5);
}

TEST(MatrixTest, Rank) {
    Matrix<double> m1(3, 3);
    m1(0, 0) = 1;
    m1(0, 1) = 2;
    m1(0, 2) = 3;
    m1(1, 0) = 4;
    m1(1, 1) = 5;
    m1(1, 2) = 6;
    m1(2, 0) = 7;
    m1(2, 1) = 8;
    m1(2, 2) = 9;
    EXPECT_EQ(m1.rank(), 2);

    Matrix<double> m2(3, 4);
    m2(0, 0) = 1;
    m2(0, 1) = 2;
    m2(0, 2) = 3;
    m2(0, 3) = 4;
    m2(1, 0) = 5;
    m2(1, 1) = 6;
    m2(1, 2) = 7;
    m2(1, 3) = 8;
    m2(2, 0) = 9;
    m2(2, 1) = 10;
    m2(2, 2) = 11;
    m2(2, 3) = 12;
    EXPECT_EQ(m2.rank(), 3);

    Matrix<double> m3(4, 2);
    m3(0, 0) = 1;
    m3(0, 1) = 2;
    m3(1, 0) = 3;
    m3(1, 1) = 6;
    m3(2, 0) = 5;
    m3(2, 1) = 10;
    m3(3, 0) = 7;
    m3(3, 1) = 14;
    EXPECT_EQ(m3.rank(), 2);
}

// TODO: add test for solve

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
