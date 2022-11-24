cmake_minimum_required(VERSION 3.0)

project(testopenpose)

file(GLOB openpose_source
     *.cpp
     *.cu
     )
add_executable(testopenpose ${openpose_source})
target_compile_features(testopenpose PUBLIC cuda_std_11)
target_compile_options(testopenpose PUBLIC -Wall -Wfloat-conversion)
target_link_libraries(testopenpose tinytrt)
target_link_libraries(testopenpose nvinfer)
target_link_libraries(testopenpose nvinfer_plugin)
target_link_libraries(testopenpose nvparsers)
target_link_libraries(testopenpose nvonnxparser)
target_link_libraries(testopenpose nvcaffe_parser)
target_link_libraries(testopenpose ${CUDART})
target_link_libraries(testopenpose ${OpenCV_LIBRARIES})