// SPDX-License-Identifier: GPL-3.0-or-later

#include <benchmark/benchmark.h>
#include "vector.h"
#include <random>

using namespace cramer;

// Helper function to create a random vector
template<typename T>
Vector<T> createRandomVector(size_t size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<T> values(size);
    for (size_t i = 0; i < size; ++i) {
        values[i] = static_cast<T>(dis(gen));
    }
    return Vector<T>(values);
}

// Benchmark constructors
static void BM_VectorDefaultConstructor(benchmark::State& state) {
    for (auto _ : state) {
        Vector<double> v;
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_VectorDefaultConstructor);

static void BM_VectorSizeConstructor(benchmark::State& state) {
    for (auto _ : state) {
        Vector<double> v(state.range(0));
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_VectorSizeConstructor)->Range(8, 8<<10);

static void BM_VectorSizeValueConstructor(benchmark::State& state) {
    for (auto _ : state) {
        Vector<double> v(state.range(0), 1.0);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_VectorSizeValueConstructor)->Range(8, 8<<10);

static void BM_VectorCopyConstructor(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> v_copy(v);
        benchmark::DoNotOptimize(v_copy);
    }
}
BENCHMARK(BM_VectorCopyConstructor)->Range(8, 8<<10);

// Benchmark size() method
static void BM_VectorSize(benchmark::State& state) {
    Vector<double> v(state.range(0));
    for (auto _ : state) {
        benchmark::DoNotOptimize(v.size());
    }
}
BENCHMARK(BM_VectorSize)->Range(8, 8<<10);

// Benchmark operator[] (both const and non-const)
static void BM_VectorOperatorBrackets(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        for (size_t i = 0; i < v.size(); ++i) {
            benchmark::DoNotOptimize(v[i]);
        }
    }
}
BENCHMARK(BM_VectorOperatorBrackets)->Range(8, 8<<10);

// Benchmark arithmetic operators
static void BM_VectorAddition(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1 + v2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorAddition)->Range(8, 8<<10);

static void BM_VectorSubtraction(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1 - v2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorSubtraction)->Range(8, 8<<10);

static void BM_VectorScalarMultiplication(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    double scalar = 2.0;
    for (auto _ : state) {
        Vector<double> result = v * scalar;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorScalarMultiplication)->Range(8, 8<<10);

// Benchmark compound assignment operators
static void BM_VectorAdditionAssignment(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1;
        result += v2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorAdditionAssignment)->Range(8, 8<<10);

static void BM_VectorSubtractionAssignment(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1;
        result -= v2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorSubtractionAssignment)->Range(8, 8<<10);

static void BM_VectorScalarMultiplicationAssignment(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    double scalar = 2.0;
    for (auto _ : state) {
        Vector<double> result = v;
        result *= scalar;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorScalarMultiplicationAssignment)->Range(8, 8<<10);

// Benchmark mathematical operations
static void BM_VectorDotProduct(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        double result = v1.dot(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorDotProduct)->Range(8, 8<<10);

static void BM_VectorNorm(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        double result = v.norm();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorNorm)->Range(8, 8<<10);

static void BM_VectorNormalize(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v.normalize();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorNormalize)->Range(8, 8<<10);

static void BM_VectorCrossProduct(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(3);
    Vector<double> v2 = createRandomVector<double>(3);
    for (auto _ : state) {
        Vector<double> result = v1.cross(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorCrossProduct);

static void BM_VectorAngle(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        double result = v1.angle(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorAngle)->Range(8, 8<<10);

static void BM_VectorProject(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1.project(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorProject)->Range(8, 8<<10);

static void BM_VectorReject(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1.reject(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorReject)->Range(8, 8<<10);

static void BM_VectorReflect(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    Vector<double> normal = createRandomVector<double>(state.range(0)).normalize();
    for (auto _ : state) {
        Vector<double> result = v.reflect(normal);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorReflect)->Range(8, 8<<10);

// Benchmark comparison operators
static void BM_VectorEquality(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = v1;
    for (auto _ : state) {
        bool result = (v1 == v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorEquality)->Range(8, 8<<10);

static void BM_VectorInequality(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        bool result = (v1 != v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorInequality)->Range(8, 8<<10);

// Benchmark element-wise operations
static void BM_VectorElementwiseMultiply(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1.elementwise_multiply(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorElementwiseMultiply)->Range(8, 8<<10);

static void BM_VectorElementwiseDivide(benchmark::State& state) {
    Vector<double> v1 = createRandomVector<double>(state.range(0));
    Vector<double> v2 = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v1.elementwise_divide(v2);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorElementwiseDivide)->Range(8, 8<<10);

// Benchmark aggregate operations
static void BM_VectorSum(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        double result = v.sum();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorSum)->Range(8, 8<<10);

static void BM_VectorProduct(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        double result = v.product();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorProduct)->Range(8, 8<<10);

static void BM_VectorMin(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        double result = v.min();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorMin)->Range(8, 8<<10);

static void BM_VectorMax(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        double result = v.max();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorMax)->Range(8, 8<<10);

// Benchmark element-wise mathematical functions
static void BM_VectorAbs(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v.abs();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorAbs)->Range(8, 8<<10);

static void BM_VectorPow(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    double exponent = 2.0;
    for (auto _ : state) {
        Vector<double> result = v.pow(exponent);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorPow)->Range(8, 8<<10);

static void BM_VectorSqrt(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v.sqrt();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorSqrt)->Range(8, 8<<10);

static void BM_VectorExp(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v.exp();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorExp)->Range(8, 8<<10);

static void BM_VectorLog(benchmark::State& state) {
    Vector<double> v = createRandomVector<double>(state.range(0));
    for (auto _ : state) {
        Vector<double> result = v.log();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorLog)->Range(8, 8<<10);
