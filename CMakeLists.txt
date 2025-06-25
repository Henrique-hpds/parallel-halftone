cmake_minimum_required(VERSION 3.18)
project(FloydSteinbergHalftone LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Includes
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA architectures (T4 = compute_75)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# ======================
# Executáveis
# ======================

# Serial C++
add_executable(fsh_serial_cpp src/serial.cpp lib/ppm.cpp)
target_link_libraries(fsh_serial_cpp PRIVATE OpenMP::OpenMP_CXX)

# Serial C
add_executable(fsh_serial_c src/serial.c)
target_link_libraries(fsh_serial_c PRIVATE ${OpenCV_LIBS} m)

# OpenMP
add_executable(fsh_omp src/omp.cpp lib/ppm.cpp)
target_link_libraries(fsh_omp PRIVATE OpenMP::OpenMP_CXX)

# CUDA
add_executable(fsh_cuda src/cuda.cu lib/ppm.cpp)
target_link_libraries(fsh_cuda PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})
set_target_properties(fsh_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
