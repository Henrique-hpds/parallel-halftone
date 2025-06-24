cmake_minimum_required(VERSION 3.18)
project(FloydSteinbergHalftone LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Serial executável (C++)
add_executable(floydsteinberg_serial src/serial.cpp)
target_link_libraries(floydsteinberg_serial PRIVATE OpenMP::OpenMP_CXX)

# Serial executável (C)
add_executable(serial src/serial.c)
target_link_libraries(serial ${OpenCV_LIBS} m)

# CUDA architectures (T4 = compute_75)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# CUDA executável principal (deve gerar cuda_dithering)
add_executable(cuda_dithering src/parallel.cu src/ppm.cpp)
target_link_libraries(cuda_dithering ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
set_target_properties(cuda_dithering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
