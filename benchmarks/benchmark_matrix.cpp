// SPDX-License-Identifier: GPL-3.0-or-later

#include <benchmark/benchmark.h>

#include <complex>
#include <random>

#include "matrix.hpp"

using namespace cramer;

// Helper function to create a random matrix
template <typename T>
Matrix<T> createRandomMatrix(size_t rows, size_t cols) {
    static std::random_device rand;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);

    Matrix<T> matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = static_cast<T>(dis(gen));
        }
    }
    return matrix;
}

// Benchmark Matrix Constructor
static void BM_MatrixConstructor(benchmark::State& state) {
    const size_t size = state.range(0);
    for (auto iter : state) {
        Matrix<double> matrix(size, size);
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_MatrixConstructor)->Range(8, 1024);

// Benchmark Matrix Addition
static void BM_MatrixAddition(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix1 = createRandomMatrix<double>(size, size);
    auto matrix2 = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixAddition)->Range(8, 1024);

// Benchmark Matrix Subtraction
static void BM_MatrixSubtraction(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix1 = createRandomMatrix<double>(size, size);
    auto matrix2 = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix1 - matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixSubtraction)->Range(8, 1024);

// Benchmark Matrix Multiplication
static void BM_MatrixMultiplication(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix1 = createRandomMatrix<double>(size, size);
    auto matrix2 = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixMultiplication)->Range(8, 256);

// Benchmark Matrix Scalar Multiplication
static void BM_MatrixScalarMultiplication(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    double scalar = 2.0;
    for (auto iter : state) {
        auto result = matrix * 2.0;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixScalarMultiplication)->Range(8, 1024);

// Benchmark Matrix Transpose
static void BM_MatrixTranspose(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix.transpose();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixTranspose)->Range(8, 1024);

// Benchmark Matrix Determinant
static void BM_MatrixDeterminant(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix.det();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixDeterminant)->Range(2, 128);

// Benchmark Matrix Inverse
static void BM_MatrixInverse(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix.inverse();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixInverse)->Range(2, 128);

// Benchmark Matrix LU Decomposition
static void BM_MatrixLU(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto [L, U] = matrix.lu();
        benchmark::DoNotOptimize(L);
        benchmark::DoNotOptimize(U);
    }
}
BENCHMARK(BM_MatrixLU)->Range(2, 128);

// Benchmark Matrix QR Decomposition
static void BM_MatrixQR(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto [Q, R] = matrix.qr();
        benchmark::DoNotOptimize(Q);
        benchmark::DoNotOptimize(R);
    }
}
BENCHMARK(BM_MatrixQR)->Range(2, 128);

// Benchmark Matrix SVD
static void BM_MatrixSVD(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto [U, S, V] = matrix.svd();
        benchmark::DoNotOptimize(U);
        benchmark::DoNotOptimize(S);
        benchmark::DoNotOptimize(V);
    }
}
BENCHMARK(BM_MatrixSVD)->Range(2, 64);

// Benchmark Matrix Eigenvalues
static void BM_MatrixEigenvalues(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix.eigenvalues();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixEigenvalues)->Range(2, 64);

// Benchmark Matrix Eigenvectors
static void BM_MatrixEigenvectors(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix.eigenvectors();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixEigenvectors)->Range(2, 64);

// Benchmark Matrix Rank
static void BM_MatrixRank(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix.rank();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixRank)->Range(2, 128);

// Benchmark Matrix Trace
static void BM_MatrixTrace(benchmark::State& state) {
    const size_t size = state.range(0);
    auto matrix = createRandomMatrix<double>(size, size);
    for (auto iter : state) {
        auto result = matrix.trace();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixTrace)->Range(8, 1024);

BENCHMARK_MAIN();
