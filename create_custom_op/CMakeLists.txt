# Inspired from https://github.com/onnx/tutorials/blob/ae0202ea5431f67ecfac03afc9987d67581f2809/PyTorchCustomOperator/ort_custom_op/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(HEADER custom_op_library.h)
set(SOURCE custom_op_library.cc)
add_library(customop SHARED ${SOURCE} ${HEADER})

# include folder in the ORT source (or find the corresponding location in release versions?)
set(ORT_INCLUDE_DIR "/home/supun/work/onnxruntime/include/")

#Include path to header files for Custom Op
# can include paths to any other location as well
# include_directories(<PATH_TO_EIGEN_DIR>)
include_directories(${ORT_INCLUDE_DIR})
# to make it possible to simply include "onnxruntime_c_api.h", cxx_api, etc.
include_directories("${ORT_INCLUDE_DIR}/onnxruntime/core/session/")

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/home/supun/work/onnxruntime/include/")
# find_library(ONNXRUNTIME_LIBRARY onnxruntime)
# target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
target_link_libraries(customop PUBLIC)