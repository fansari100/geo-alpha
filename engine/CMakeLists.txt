# geo-alpha state-space engine.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --parallel
#   ctest --test-dir build --output-on-failure
#
# Optional: -DGEOALPHA_BUILD_PYTHON=ON to build the pybind11 module.

cmake_minimum_required(VERSION 3.20)
project(geoalpha_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(GEOALPHA_BUILD_TESTS "Build GoogleTest tests" ON)
option(GEOALPHA_BUILD_PYTHON "Build pybind11 Python bindings" OFF)

add_library(geoalpha_engine SHARED
    src/linalg.cpp
    src/kalman.cpp
    src/ukf.cpp
)
target_include_directories(geoalpha_engine PUBLIC include)

if(MSVC)
    target_compile_options(geoalpha_engine PRIVATE /W4 /O2 /fp:fast)
else()
    target_compile_options(geoalpha_engine PRIVATE -Wall -Wextra -Wpedantic -O3 -ffast-math)
endif()

# Standalone benchmark harness.
add_executable(geoalpha_bench bench/bench.cpp)
target_link_libraries(geoalpha_bench PRIVATE geoalpha_engine)

if(GEOALPHA_BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()
        add_executable(geoalpha_tests
            tests/test_linalg.cpp
            tests/test_kalman.cpp
            tests/test_ukf.cpp
        )
        target_link_libraries(geoalpha_tests PRIVATE
            geoalpha_engine GTest::gtest GTest::gtest_main)
        include(GoogleTest)
        gtest_discover_tests(geoalpha_tests)
    endif()
endif()

if(GEOALPHA_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(geoalpha_engine_py bindings/bindings.cpp)
    target_link_libraries(geoalpha_engine_py PRIVATE geoalpha_engine)
endif()

install(TARGETS geoalpha_engine LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
