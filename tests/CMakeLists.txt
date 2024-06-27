# Minimum CMake version
cmake_minimum_required(VERSION 3.12)

# Find Google Test
find_package(GTest REQUIRED)

# Include directories
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable
add_executable(test_matrix test_matrix.cpp)

# Link libraries
target_link_libraries(test_matrix
    PRIVATE
        cramer
        GTest::GTest
        GTest::Main
)

# Add the test to CTest
add_test(NAME MatrixTest COMMAND test_matrix)

# Set C++ standard for the test
set_target_properties(test_matrix PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Apply address sanitizer flags
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=address")
    set_target_properties(test_matrix PROPERTIES
        COMPILE_FLAGS "${CMAKE_CXX_FLAGS}"
        LINK_FLAGS "-fsanitize=address"
    )
endif()
