cmake_minimum_required(VERSION 3.15) 
PROJECT ( main )
find_package(OpenCV REQUIRED )


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src/c++
        # ${PROJECT_SOURCE_DIR}/testing

)

file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/include/c++/*.h"
        "${PROJECT_SOURCE_DIR}/include/c++/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/c++/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/c++/*.c"
        # "${PROJECT_SOURCE_DIR}/testing/*.cpp"
        # "${PROJECT_SOURCE_DIR}/testing/*.hpp"
        )


add_executable( main ${all_SRCS} )

# add_library(main SHARED ${all_SRCS})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )