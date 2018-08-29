cmake_minimum_required(VERSION 3.1)
project(quadboost VERSION 0.1.0)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

add_executable(test ./weak_learner/decision_stump/test_xtensor.cc)

target_link_libraries(test xtensor)
