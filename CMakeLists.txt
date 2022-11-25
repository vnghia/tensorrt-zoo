cmake_minimum_required(VERSION 3.22)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_compile_options(-fcolor-diagnostics)
endif()

# Project
project(
  openposert
  VERSION 0.0.1
  LANGUAGES CXX
)
enable_language(CUDA)
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_ROOT_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Prevent building in the source directory
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
    "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# Ccache
option(ENABLE_CCACHE "Automatically use ccache if available" ON)

if("${ENABLE_CCACHE}")
  find_program(CCACHE_PROGRAM ccache NAMES /usr/bin/ccache)

  if(CCACHE_PROGRAM)
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CC_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

# Output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_ROOT_DIR}/lib/${CMAKE_BUILD_TYPE}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_ROOT_DIR}/lib/${CMAKE_BUILD_TYPE}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_ROOT_DIR}/bin/${CMAKE_BUILD_TYPE}")

# Tiny TensorRT
include(external/tiny_tensorrt)

# OpenCV
find_package(OpenCV REQUIRED)

message(STATUS "
=============> Final Config <===============
TensorRT headers : ${TENSORRT_INCLUDE_DIR}
TensorRT library : ${LIBNVINFER}
CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}
============================================
")

# OpenposeRT
add_library(${PROJECT_NAME}
  src/array.cpp
  src/body_part_connector.cu
  src/open_pose.cu
  src/point.cpp
  src/pose_nms.cu
  src/resize_and_merge.cu
)

target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_17)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_link_libraries(${PROJECT_NAME}
  PUBLIC ${OpenCV_LIBS}
  tinytrt
)

target_link_directories(${PROJECT_NAME} PUBLIC ${OpenCV_LIB_DIR})

target_include_directories(${PROJECT_NAME}
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
