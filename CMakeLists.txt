cmake_minimum_required(VERSION 3.12)
project(cramer VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AddressSanitizer and Coverage
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address -fprofile-arcs -ftest-coverage")

# Find dependencies
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(
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
