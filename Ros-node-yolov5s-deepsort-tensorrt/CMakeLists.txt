cmake_minimum_required(VERSION 2.6)

project(yolov5ssort)
SET(CMAKE_BUILD_TYPE Debug)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++0x")
set(OpenCV_DIR /home/swh/opencv-4.5.0/build)
set(cv_bridge_DIR /home/swh/cv_bridge_ws/devel/share/cv_bridge/cmake)
find_package(OpenCV REQUIRED)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)



find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_53;code=sm_53)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/deepsort/include)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/swh/TensorRT-7.2.3.4/include/)
link_directories( /home/swh/TensorRT-7.2.3.4/lib)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/deepsort/include
)
aux_source_directory(${PROJECT_SOURCE_DIR}/deepsort/src SRC_DIR)

# ===== deepsort =====
add_library(deepsort SHARED ${SRC_DIR})
target_link_libraries(deepsort 
    ${CUDA_LIBS} ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser
)

# ===== yolo =====
SET(CMAKE_BUILD_TYPE Debug)
include_directories(${PROJECT_SOURCE_DIR}/yolo/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/yolo/src YOLO_SRC_DIR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolov5s_trt SHARED ${PROJECT_SOURCE_DIR}/yolo/include/yololayer.cu ${PROJECT_SOURCE_DIR}/yolo/src/yolov5s_lib.cpp)
target_link_libraries(yolov5s_trt nvinfer cudart deepsort) 


# ===== main =====
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  roscpp
  rospy
  cv_bridge
  image_transport
  sensor_msgs
  can_control_msgs
)

message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    LIBRARIES: ${OpenCV_LIBRARIES}")
message(STATUS "    catkin_LIBRARIES: ${catkin_LIBRARIES}")
catkin_package(
  CATKIN_DEPENDS
  message_runtime
 INCLUDE_DIRS include
#  LIBRARIES can_com
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)
include_directories(
  include 
  ${catkin_INCLUDE_DIRS}
  ${COMMON_INCLUDE}
# include
# ${catkin_INCLUDE_DIRS}
)


aux_source_directory(${PROJECT_SOURCE_DIR}/src M_SRC_DIR)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${catkin_LIB_DIRS}   ${PROJECT_SOURCE_DIR}/lib)
add_executable(yolov5ssort ${M_SRC_DIR})
#add_dependencies(yolov5ssort  ${catkin_EXPORTED_TARGETS})


target_link_libraries(yolov5ssort ${catkin_LIBRARIES}  nvinfer cudart yolov5s_trt)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)
set(CMAKE_CXX_FLAGS "-std=c++0x")
