cmake_minimum_required(VERSION 3.10)
project(MC970FinalProject)

# C++ padrão
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-c++17-extensions")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------------
# Versão serial
add_executable(serial src/serial.c)
target_link_libraries(serial ${OpenCV_LIBS} m)

# ------------------------
# Versão CUDA (main + CUDA kernel)
add_executable(cuda_dithering
    # src/main.cpp
    src/parallel.cu
)
target_link_libraries(cuda_dithering ${OpenCV_LIBS})
set_target_properties(cuda_dithering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)