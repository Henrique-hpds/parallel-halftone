cmake_minimum_required(VERSION 3.18)
project(FloydSteinbergHalftone LANGUAGES CXX CUDA)

# Configuração geral
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

# Executável Serial (compila apenas com C++)
add_executable(floydsteinberg_serial serial.cpp)
target_link_libraries(floydsteinberg_serial PRIVATE OpenMP::OpenMP_CXX)

<<<<<<< HEAD
# CUDA architectures (T4 = compute_75)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

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
=======
# Executável CUDA Paralelo
add_executable(floydsteinberg_parallel parallel.cu)
target_link_libraries(floydsteinberg_parallel PRIVATE OpenMP::OpenMP_CXX)
>>>>>>> origin/cuda2
