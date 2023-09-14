cmake_minimum_required(VERSION 3.5)
project(Mantis_Simulation)

FIND_PACKAGE( OpenCV REQUIRED )


set(SOURCE_FILES Mantis_Simulation.cpp)

add_executable(Mantis_Simulation ${SOURCE_FILES})

target_link_libraries(Mantis_Simulation ${OpenCV_LIBS} )