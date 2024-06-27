cmake_minimum_required(VERSION 3.12)
project(cramer VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always enable code coverage
add_compile_options(--coverage -g -O0)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

# Enable AddressSanitizer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")

# Find dependencies
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(
    ${GTEST_INCLUDE_DIRS}
    include
)

# Cramer library
add_library(cramer STATIC
    src/matrix.cpp
    src/vector.cpp
)
target_include_directories(cramer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cramer PRIVATE
    Threads::Threads
)

# Enable testing
enable_testing()

# Add the tests subdirectory
add_subdirectory(tests)

# Benchmarking
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.6.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Add combined benchmark executable
add_executable(benchmark_all 
    benchmarks/benchmark_main.cpp
    benchmarks/benchmark_vector.cpp
    benchmarks/benchmark_matrix.cpp
)
target_include_directories(benchmark_all PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)
target_link_libraries(benchmark_all PRIVATE 
    cramer 
    benchmark::benchmark 
)

# Make sure benchmarks are built by default
add_custom_target(build_all ALL DEPENDS cramer benchmark_all)
