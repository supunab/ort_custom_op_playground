cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(HEADER custom_op_library.h)
set(SOURCE custom_op_library.cc)
add_library(customop SHARED ${SOURCE} ${HEADER})

#Include path to header files for Custom Op
# can include paths to any other location as well
# include_directories(<PATH_TO_EIGEN_DIR>)
include_directories("/home/supun/work/onnxruntime/include/")


#Include path to header files for Custom Op Test
# include_directories(<PATH_TO_ONNXRUNTIME_TEST_UTIL_INCLUDE_DIR>)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/home/supun/work/onnxruntime/include/")
# find_library(ONNXRUNTIME_LIBRARY onnxruntime)
# target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
target_link_libraries(customop PUBLIC)