# geo-alpha state-space engine.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --parallel
#   ctest --test-dir build --output-on-failure
#
# Optional:
#   -DGEOALPHA_BUILD_PYTHON=ON   build the pybind11 Python module
#   -DGEOALPHA_BUILD_TESTS=OFF   skip GoogleTest binaries
#   -DGEOALPHA_USE_FETCHED_GTEST=ON  always FetchContent gtest (no system search)

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
option(GEOALPHA_USE_FETCHED_GTEST
       "Always download GoogleTest via FetchContent rather than find_package" OFF)

# Built as STATIC by default so the test and bench binaries do not require
# runtime resolution of the shared-library RPATH on the CI runner. Set
# -DBUILD_SHARED_LIBS=ON to produce a shared library for distribution.
add_library(geoalpha_engine
    src/linalg.cpp
    src/kalman.cpp
    src/ukf.cpp
)
target_include_directories(geoalpha_engine PUBLIC include)

if(MSVC)
    target_compile_options(geoalpha_engine PRIVATE /W4 /O2 /fp:fast)
else()
    target_compile_options(geoalpha_engine PRIVATE -Wall -Wextra -O2)
endif()

# Standalone benchmark harness.
add_executable(geoalpha_bench bench/bench.cpp)
target_link_libraries(geoalpha_bench PRIVATE geoalpha_engine)

# --------------------------------------------------------------------- #
# GoogleTest discovery.
#
# libgtest-dev packages expose different CMake target names depending on
# distribution and CMake version:
#   Ubuntu / Debian (recent):    GTest::gtest, GTest::gtest_main
#   CMake's bundled FindGTest:   GTest::GTest, GTest::Main
#   Older / source-only builds:  GTEST_LIBRARIES + GTEST_INCLUDE_DIRS
# All three target sets are checked, with FetchContent v1.14.0 as the
# final fallback.
# --------------------------------------------------------------------- #
if(GEOALPHA_BUILD_TESTS)
    set(_GA_GTEST_TARGETS "")

    if(NOT GEOALPHA_USE_FETCHED_GTEST)
        find_package(GTest QUIET)
    endif()

    if(GTest_FOUND AND TARGET GTest::gtest AND TARGET GTest::gtest_main)
        set(_GA_GTEST_TARGETS GTest::gtest GTest::gtest_main)
        message(STATUS "geo-alpha: using system GTest::gtest / GTest::gtest_main")
    elseif(GTest_FOUND AND TARGET GTest::GTest AND TARGET GTest::Main)
        set(_GA_GTEST_TARGETS GTest::GTest GTest::Main)
        message(STATUS "geo-alpha: using system GTest::GTest / GTest::Main")
    else()
        message(STATUS "geo-alpha: fetching GoogleTest via FetchContent")
        include(FetchContent)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
        set(_GA_GTEST_TARGETS gtest gtest_main)
    endif()

    enable_testing()
    add_executable(geoalpha_tests
        tests/test_linalg.cpp
        tests/test_kalman.cpp
        tests/test_ukf.cpp
    )
    target_link_libraries(geoalpha_tests PRIVATE
        geoalpha_engine ${_GA_GTEST_TARGETS})

    include(GoogleTest)
    gtest_discover_tests(geoalpha_tests)
endif()

if(GEOALPHA_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(geoalpha_engine_py bindings/bindings.cpp)
    target_link_libraries(geoalpha_engine_py PRIVATE geoalpha_engine)
endif()

install(TARGETS geoalpha_engine
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
