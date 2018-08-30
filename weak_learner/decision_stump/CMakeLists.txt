cmake_minimum_required(VERSION 3.1)
project(quadboost VERSION 0.1.0)


# set(xtl_DIR )
find_package(xtl REQUIRED PATHS "C:/Users/Jean-Samuel/Miniconda3/Library/lib/cmake")
find_package(xtensor REQUIRED PATHS "C:/Users/Jean-Samuel/Miniconda3/Library/lib/cmake")

include_directories(src)
file(GLOB SOURCES "src/*.cc")

add_executable(find_stump_test ${SOURCES})

target_link_libraries(find_stump_test xtensor)
