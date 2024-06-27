// SPDX-License-Identifier: GPL-3.0-or-later

#include <benchmark/benchmark.h>
#include "matrix.h"
#include <random>
#include <complex>

using namespace cramer;

// Helper function to create a random matrix
template<typename T>
Matrix<T> createRandomMatrix(size_t rows, size_t cols) {
    static std::random_device rd;
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
    for (auto _ : state) {
        Matrix<double> m(size, size);
        benchmark::DoNotOptimize(m);
    }
}
BENCHMARK(BM_MatrixConstructor)->Range(8, 1024);

// Benchmark Matrix Addition
static void BM_MatrixAddition(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m1 = createRandomMatrix<double>(size, size);
    auto m2 = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m1 + m2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixAddition)->Range(8, 1024);

// Benchmark Matrix Subtraction
static void BM_MatrixSubtraction(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m1 = createRandomMatrix<double>(size, size);
    auto m2 = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m1 - m2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixSubtraction)->Range(8, 1024);

// Benchmark Matrix Multiplication
static void BM_MatrixMultiplication(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m1 = createRandomMatrix<double>(size, size);
    auto m2 = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m1 * m2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixMultiplication)->Range(8, 256);

// Benchmark Matrix Scalar Multiplication
static void BM_MatrixScalarMultiplication(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    double scalar = 2.0;
    for (auto _ : state) {
        auto result = m * scalar;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixScalarMultiplication)->Range(8, 1024);

// Benchmark Matrix Transpose
static void BM_MatrixTranspose(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m.transpose();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixTranspose)->Range(8, 1024);

// Benchmark Matrix Determinant
static void BM_MatrixDeterminant(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m.det();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixDeterminant)->Range(2, 128);

// Benchmark Matrix Inverse
static void BM_MatrixInverse(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m.inverse();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixInverse)->Range(2, 128);

// Benchmark Matrix LU Decomposition
static void BM_MatrixLU(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto [L, U] = m.lu();
        benchmark::DoNotOptimize(L);
        benchmark::DoNotOptimize(U);
    }
}
BENCHMARK(BM_MatrixLU)->Range(2, 128);

// Benchmark Matrix QR Decomposition
static void BM_MatrixQR(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto [Q, R] = m.qr();
        benchmark::DoNotOptimize(Q);
        benchmark::DoNotOptimize(R);
    }
}
BENCHMARK(BM_MatrixQR)->Range(2, 128);

// Benchmark Matrix SVD
static void BM_MatrixSVD(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto [U, S, V] = m.svd();
        benchmark::DoNotOptimize(U);
        benchmark::DoNotOptimize(S);
        benchmark::DoNotOptimize(V);
    }
}
BENCHMARK(BM_MatrixSVD)->Range(2, 64);

// Benchmark Matrix Eigenvalues
static void BM_MatrixEigenvalues(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m.eigenvalues();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixEigenvalues)->Range(2, 64);

// Benchmark Matrix Eigenvectors
static void BM_MatrixEigenvectors(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m.eigenvectors();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixEigenvectors)->Range(2, 64);

// Benchmark Matrix Rank
static void BM_MatrixRank(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m.rank();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixRank)->Range(2, 128);

// Benchmark Matrix Trace
static void BM_MatrixTrace(benchmark::State& state) {
    const size_t size = state.range(0);
    auto m = createRandomMatrix<double>(size, size);
    for (auto _ : state) {
        auto result = m.trace();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixTrace)->Range(8, 1024);
