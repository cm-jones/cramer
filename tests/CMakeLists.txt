include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Test sources
file(GLOB TEST_SOURCES *.cpp)

# Create the test executable
add_executable(runTests ${TEST_SOURCES})

# Link Google Test and your library
target_link_libraries(runTests gtest gtest_main lincpp)

# Register tests
include(GoogleTest)
gtest_discover_tests(runTests)
