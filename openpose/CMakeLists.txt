cmake_minimum_required(VERSION 3.22)

# Project
project(testopenpose)

file(GLOB openpose_source
  *.cpp
  *.cu
)

add_executable(testopenpose ${openpose_source})
target_compile_features(testopenpose PUBLIC cuda_std_11)
target_compile_options(testopenpose PUBLIC -Wall -Wfloat-conversion)
target_include_directories(testopenpose PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_directories(testopenpose PUBLIC ${OpenCV_LIB_DIR})
target_link_libraries(testopenpose PUBLIC
  ${OpenCV_LIBS}
  tinytrt
)
