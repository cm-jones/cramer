// SPDX-License-Identifier: GPL-3.0-or-later

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "vector.hpp"

using namespace cramer;

class VectorTest : public ::testing::Test {
protected:
    // Helper function to compare floating point numbers
    bool is_close(double first, double second, double tolerance = 1e-9) {
        return std::abs(first - second) < tolerance;
    }

    // Helper function to compare vectors
    template <typename T>
    bool vectors_are_close(const Vector<T>& first, const Vector<T>& second,
                           double tolerance = 1e-9) {
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

// Test constructors
TEST_F(VectorTest, DefaultConstructor) {
    Vector<double> vec;
    EXPECT_EQ(vec.size(), 0);
}

TEST_F(VectorTest, SizeConstructor) {
    Vector<double> vec(3);
    EXPECT_EQ(vec.size(), 3);
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(vec[i], 0.0);
    }
}

TEST_F(VectorTest, SizeValueConstructor) {
    Vector<double> vec(3, 2.5);
    EXPECT_EQ(vec.size(), 3);
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(vec[i], 2.5);
    }
}

TEST_F(VectorTest, VectorConstructor) {
    std::vector<double> std_vec = {1.0, 2.0, 3.0};
    Vector<double> vec(std_vec);
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 1.0);
    EXPECT_EQ(vec[1], 2.0);
    EXPECT_EQ(vec[2], 3.0);
}

// Test element access
TEST_F(VectorTest, ElementAccess) {
    Vector<double> vec(3, 1.0);
    vec[1] = 2.0;
    EXPECT_EQ(vec[0], 1.0);
    EXPECT_EQ(vec[1], 2.0);
    EXPECT_EQ(vec[2], 1.0);
}

TEST_F(VectorTest, ConstElementAccess) {
    const Vector<double> vec(std::vector<double>{1.0, 2.0, 3.0});
    EXPECT_EQ(vec[0], 1.0);
    EXPECT_EQ(vec[1], 2.0);
    EXPECT_EQ(vec[2], 3.0);
}

// Test operators
TEST_F(VectorTest, AdditionOperator) {
    Vector<double> vec1(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec2(std::vector<double>{4.0, 5.0, 6.0});
    Vector<double> result = vec1 + vec2;
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 5.0);
    EXPECT_EQ(result[1], 7.0);
    EXPECT_EQ(result[2], 9.0);
}

TEST_F(VectorTest, AdditionAssignmentOperator) {
    Vector<double> vec1(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec2(std::vector<double>{4.0, 5.0, 6.0});
    vec1 += vec2;
    EXPECT_EQ(vec1.size(), 3);
    EXPECT_EQ(vec1[0], 5.0);
    EXPECT_EQ(vec1[1], 7.0);
    EXPECT_EQ(vec1[2], 9.0);
}

TEST_F(VectorTest, SubtractionOperator) {
    Vector<double> vec1(std::vector<double>{4.0, 5.0, 6.0});
    Vector<double> vec2(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> result = vec1 - vec2;
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 3.0);
    EXPECT_EQ(result[1], 3.0);
    EXPECT_EQ(result[2], 3.0);
}

TEST_F(VectorTest, SubtractionAssignmentOperator) {
    Vector<double> vec1(std::vector<double>{4.0, 5.0, 6.0});
    Vector<double> vec2(std::vector<double>{1.0, 2.0, 3.0});
    vec1 -= vec2;
    EXPECT_EQ(vec1.size(), 3);
    EXPECT_EQ(vec1[0], 3.0);
    EXPECT_EQ(vec1[1], 3.0);
    EXPECT_EQ(vec1[2], 3.0);
}

TEST_F(VectorTest, ScalarMultiplicationOperator) {
    Vector<double> vec(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> result = vec * 2.0;
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 2.0);
    EXPECT_EQ(result[1], 4.0);
    EXPECT_EQ(result[2], 6.0);
}

TEST_F(VectorTest, ScalarMultiplicationAssignmentOperator) {
    Vector<double> vec(std::vector<double>{1.0, 2.0, 3.0});
    vec *= 2.0;
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 2.0);
    EXPECT_EQ(vec[1], 4.0);
    EXPECT_EQ(vec[2], 6.0);
}

TEST_F(VectorTest, EqualityOperator) {
    Vector<double> vec1(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec2(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec3(std::vector<double>{1.0, 2.0, 4.0});
    EXPECT_TRUE(vec1 == vec2);
    EXPECT_FALSE(vec1 == vec3);
}

TEST_F(VectorTest, InequalityOperator) {
    Vector<double> vec1(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec2(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec3(std::vector<double>{1.0, 2.0, 4.0});
    EXPECT_FALSE(vec1 != vec2);
    EXPECT_TRUE(vec1 != vec3);
}

// Test vector operations
TEST_F(VectorTest, DotProduct) {
    Vector<double> vec1(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec2(std::vector<double>{4.0, 5.0, 6.0});
    double result = vec1.dot(vec2);
    EXPECT_DOUBLE_EQ(result, 32.0);  // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(VectorTest, Norm) {
    Vector<double> vec(std::vector<double>{3.0, 4.0});
    EXPECT_DOUBLE_EQ(vec.norm(), 5.0);  // sqrt(3^2 + 4^2) = 5
}

TEST_F(VectorTest, Normalize) {
    Vector<double> vec(std::vector<double>{3.0, 4.0});
    Vector<double> normalized = vec.normalize();
    EXPECT_DOUBLE_EQ(normalized[0], 0.6);  // 3/5 = 0.6
    EXPECT_DOUBLE_EQ(normalized[1], 0.8);  // 4/5 = 0.8
    EXPECT_DOUBLE_EQ(normalized.norm(), 1.0);
}

TEST_F(VectorTest, NormalizeZeroVector) {
    Vector<double> vec(3, 0.0);
    EXPECT_THROW(vec.normalize(), std::runtime_error);
}

TEST_F(VectorTest, CrossProduct) {
    Vector<double> vec1(std::vector<double>{1.0, 0.0, 0.0});
    Vector<double> vec2(std::vector<double>{0.0, 1.0, 0.0});
    Vector<double> result = vec1.cross(vec2);
    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 1.0);
}

TEST_F(VectorTest, CrossProductInvalidDimension) {
    Vector<double> vec1(std::vector<double>{1.0, 0.0});
    Vector<double> vec2(std::vector<double>{0.0, 1.0});
    EXPECT_THROW(vec1.cross(vec2), std::invalid_argument);
}

TEST_F(VectorTest, Angle) {
    Vector<double> vec1(std::vector<double>{1.0, 0.0});
    Vector<double> vec2(std::vector<double>{0.0, 1.0});
    EXPECT_DOUBLE_EQ(vec1.angle(vec2), M_PI/2);  // 90 degrees = PI/2
}

TEST_F(VectorTest, AngleZeroVector) {
    Vector<double> vec1(std::vector<double>{1.0, 0.0});
    Vector<double> vec2(2, 0.0);
    EXPECT_THROW(vec1.angle(vec2), std::runtime_error);
}

TEST_F(VectorTest, Project) {
    Vector<double> vec1(std::vector<double>{3.0, 3.0});
    Vector<double> vec2(std::vector<double>{0.0, 1.0});
    Vector<double> result = vec1.project(vec2);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
}

TEST_F(VectorTest, Reject) {
    Vector<double> vec1(std::vector<double>{3.0, 3.0});
    Vector<double> vec2(std::vector<double>{0.0, 1.0});
    Vector<double> result = vec1.reject(vec2);
    EXPECT_DOUBLE_EQ(result[0], 3.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
}

TEST_F(VectorTest, Reflect) {
    Vector<double> vec(std::vector<double>{1.0, -1.0});
    Vector<double> normal(std::vector<double>{0.0, 1.0});
    Vector<double> result = vec.reflect(normal);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
}

// Test element-wise operations
TEST_F(VectorTest, ElementwiseMultiply) {
    Vector<double> vec1(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec2(std::vector<double>{4.0, 5.0, 6.0});
    Vector<double> result = vec1.elementwise_multiply(vec2);
    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 4.0);
    EXPECT_DOUBLE_EQ(result[1], 10.0);
    EXPECT_DOUBLE_EQ(result[2], 18.0);
}

TEST_F(VectorTest, ElementwiseDivide) {
    Vector<double> vec1(std::vector<double>{4.0, 10.0, 18.0});
    Vector<double> vec2(std::vector<double>{2.0, 5.0, 6.0});
    Vector<double> result = vec1.elementwise_divide(vec2);
    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST_F(VectorTest, ElementwiseDivideByZero) {
    Vector<double> vec1(std::vector<double>{1.0, 2.0, 3.0});
    Vector<double> vec2(std::vector<double>{1.0, 0.0, 3.0});
    EXPECT_THROW(vec1.elementwise_divide(vec2), std::runtime_error);
}

// Test aggregate operations
TEST_F(VectorTest, Sum) {
    Vector<double> vec(std::vector<double>{1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(vec.sum(), 6.0);
}

TEST_F(VectorTest, Product) {
    Vector<double> vec(std::vector<double>{2.0, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(vec.product(), 24.0);
}

TEST_F(VectorTest, Min) {
    Vector<double> vec(std::vector<double>{3.0, 1.0, 2.0});
    EXPECT_DOUBLE_EQ(vec.min(), 1.0);
}

TEST_F(VectorTest, MinEmptyVector) {
    Vector<double> vec;
    EXPECT_THROW(vec.min(), std::runtime_error);
}

TEST_F(VectorTest, Max) {
    Vector<double> vec(std::vector<double>{3.0, 1.0, 2.0});
    EXPECT_DOUBLE_EQ(vec.max(), 3.0);
}

TEST_F(VectorTest, MaxEmptyVector) {
    Vector<double> vec;
    EXPECT_THROW(vec.max(), std::runtime_error);
}

// Test element-wise math operations
TEST_F(VectorTest, Abs) {
    Vector<double> vec(std::vector<double>{-1.0, 2.0, -3.0});
    Vector<double> result = vec.abs();
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST_F(VectorTest, Pow) {
    Vector<double> vec(std::vector<double>{2.0, 3.0, 4.0});
    Vector<double> result = vec.pow(2.0);
    EXPECT_DOUBLE_EQ(result[0], 4.0);
    EXPECT_DOUBLE_EQ(result[1], 9.0);
    EXPECT_DOUBLE_EQ(result[2], 16.0);
}

TEST_F(VectorTest, Sqrt) {
    Vector<double> vec(std::vector<double>{4.0, 9.0, 16.0});
    Vector<double> result = vec.sqrt();
    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[2], 4.0);
}

TEST_F(VectorTest, SqrtNegative) {
    Vector<double> vec(std::vector<double>{4.0, -9.0, 16.0});
    EXPECT_THROW(vec.sqrt(), std::runtime_error);
}

TEST_F(VectorTest, Exp) {
    Vector<double> vec(std::vector<double>{0.0, 1.0, 2.0});
    Vector<double> result = vec.exp();
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_NEAR(result[1], 2.718281828459045, 1e-9);
    EXPECT_NEAR(result[2], 7.38905609893065, 1e-9);
}

TEST_F(VectorTest, Log) {
    Vector<double> vec(std::vector<double>{1.0, 2.718281828459045, 7.38905609893065});
    Vector<double> result = vec.log();
    EXPECT_NEAR(result[0], 0.0, 1e-9);
    EXPECT_NEAR(result[1], 1.0, 1e-9);
    EXPECT_NEAR(result[2], 2.0, 1e-9);
}

TEST_F(VectorTest, LogNonPositive) {
    Vector<double> vec(std::vector<double>{1.0, 0.0, 2.0});
    EXPECT_THROW(vec.log(), std::runtime_error);
}

// Test complex number operations
TEST_F(VectorTest, ComplexVectorOperations) {
    using complex = std::complex<double>;
    Vector<complex> vec1(std::vector<complex>{complex(1.0, 1.0), complex(2.0, 2.0)});
    Vector<complex> vec2(std::vector<complex>{complex(3.0, 3.0), complex(4.0, 4.0)});
    
    // Test addition
    Vector<complex> sum = vec1 + vec2;
    EXPECT_EQ(sum[0], complex(4.0, 4.0));
    EXPECT_EQ(sum[1], complex(6.0, 6.0));
    
    // Test dot product
    complex dot_product = vec1.dot(vec2);
    EXPECT_EQ(dot_product, complex(0.0, 40.0));
    
    // Test norm
    EXPECT_NEAR(vec1.norm(), 3.0, 1e-9);  // sqrt(|1+1i|² + |2+2i|²) = sqrt(2 + 8) = sqrt(10) ≈ 3.16
    
    // Test element-wise operations
    Vector<complex> elem_mult = vec1.elementwise_multiply(vec2);
    EXPECT_EQ(elem_mult[0], complex(0.0, 6.0));  // (1+i)*(3+3i) = 3+3i+3i-3 = 0+6i
    EXPECT_EQ(elem_mult[1], complex(0.0, 16.0)); // (2+2i)*(4+4i) = 8+8i+8i-8 = 0+16i
}

// Test basic operations (from test_vector_full.cpp)
TEST_F(VectorTest, BasicOperations) {
    Vector<double> v1(3, 1.0);
    Vector<double> v2(3, 2.0);
    
    // Test addition
    Vector<double> sum = v1 + v2;
    for (size_t i = 0; i < sum.size(); ++i) {
        EXPECT_DOUBLE_EQ(sum[i], 3.0);
    }
    
    // Test subtraction
    Vector<double> diff = v2 - v1;
    for (size_t i = 0; i < diff.size(); ++i) {
        EXPECT_DOUBLE_EQ(diff[i], 1.0);
    }
    
    // Test scalar multiplication
    Vector<double> prod = v1 * 3.0;
    for (size_t i = 0; i < prod.size(); ++i) {
        EXPECT_DOUBLE_EQ(prod[i], 3.0);
    }
    
    // Test dot product
    EXPECT_DOUBLE_EQ(v1.dot(v2), 6.0);  // 3 * (1*2)
    
    // Test norm
    Vector<double> v3(2);
    v3[0] = 3.0;
    v3[1] = 4.0;
    EXPECT_DOUBLE_EQ(v3.norm(), 5.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
