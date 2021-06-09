cmake_minimum_required(VERSION 3.0.0)
project(CurveFitting)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(INCLUDES_DIR ${PROJECT_SOURCE_DIR}/../../include)
set(SRC_DIR ${PROJECT_SOURCE_DIR}/../../src)

find_package(Eigen3 REQUIRED)

include_directories(
	${EIGEN3_INCLUDE_DIR}
	${INCLUDES_DIR}
)

add_executable(${PROJECT_NAME}
	${SRC_DIR}/Edge.cpp
	${SRC_DIR}/LossFunction.cpp
	${SRC_DIR}/Problem.cpp
	${SRC_DIR}/Vertex.cpp
	${PROJECT_SOURCE_DIR}/CurveFitting.cpp
)