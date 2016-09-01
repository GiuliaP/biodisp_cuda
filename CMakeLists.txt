# Author: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.8)

set(PROJECTNAME biodisp_cuda)
project(${PROJECTNAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${OpenCV_INCLUDE_DIRS})

# no need to use the following instruction because this is automatically done by CUDA_ADD_EXECUTABLE
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(folder_source 
src/Filters.cpp
src/Population.cpp
src/main.cpp
src/conv.cu)

set(folder_header
include/Filters.h
include/Population.h
include/quantities.h
include/conv_common.h
include/helper_cuda.h
include/helper_string.h
include/helper_functions.h
include/exception.h
include/helper_timer.h
include/helper_image.h)

source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})

target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS ${PROJECTNAME} DESTINATION bin)
