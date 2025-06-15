cmake_minimum_required(VERSION 3.10)
project(MC970FinalProject)

# C++ padrão
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # CUDA (ativa suporte)
# enable_language(CUDA)
# set(CMAKE_CUDA_STANDARD 14)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------------
# Serial
add_executable(serial src/serial.cpp)
target_link_libraries(serial ${OpenCV_LIBS})

# # ------------------------
# # OpenMP
# find_package(OpenMP REQUIRED)
# add_executable(versao_openmp openmp.cpp)
# target_link_libraries(versao_openmp ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# # ------------------------
# # CUDA
# add_executable(versao_cuda cuda.cu)
# target_link_libraries(versao_cuda ${OpenCV_LIBS})
