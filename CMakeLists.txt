cmake_minimum_required(VERSION 3.10)
project(cramer)

set(CMAKE_CXX_STANDARD 14)

# Include directories
include_directories(include)

# Source files
file(GLOB SOURCES src/*.cpp)

# Create a static library
add_library(cramer STATIC ${SOURCES})

# Specify the include directory for the library target
target_include_directories(cramer PUBLIC include)

# Optionally, you can add an executable target
# add_executable(main main.cpp)
# target_link_libraries(main cramer)

# Installation rules
install(TARGETS cramer DESTINATION lib)
install(DIRECTORY include/numcpp DESTINATION include)
