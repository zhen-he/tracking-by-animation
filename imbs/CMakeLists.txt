project(imbs)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_BINARY_DIR})

add_executable (imbs main.cpp imbs.cpp imagemanager.cpp)
target_link_libraries (imbs ${OpenCV_LIBS})
