cmake_minimum_required (VERSION 2.8)

project (yolo_cap)
# using C++11 
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")

#if u have OpenCV version more than one, set the build path which one u want to use
# set(OpenCV_DIR "{Your_Path}build/")

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

#if use realsense,uncommit this
#set(DEPENDENCIES realsense ${OPENGL_LIBRARIES})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
#message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")

aux_source_directory(./src/ DIR_SRCS)

add_executable(yolo_cap ${DIR_SRCS})

include_directories (
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
#darknet path
    {Your_Path}/darknet/include
    {Your_Path}/darknet/src
#cuda path
    #/usr/local/cuda-9.0/include    
)

target_link_libraries(yolo_cap
${OpenCV_LIBS}
/usr/lib
/usr/local/lib
${DEPENDENCIES}
#darknet lib path
{Your_Path}/darknet/libdarknet.so
    )