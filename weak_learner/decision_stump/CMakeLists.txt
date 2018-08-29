cmake_minimum_required(VERSION 3.1)
project(quadboost VERSION 0.1.0)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

include_directories(src/)
file(GLOB SOURCES "src/*.cc")

add_executable(test_xtensor ${SOURCES})

target_link_libraries(test_xtensor xtensor)
