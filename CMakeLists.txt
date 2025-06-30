cmake_minimum_required(VERSION 3.18)
project(FloydSteinbergHalftone LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)

# Includes
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA architectures (T4 = compute_75)
set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 89 90)

# ======================
# Executáveis
# ======================

# Serial C++
add_executable(serial_cpp src/serial.cpp lib/ppm.cpp lib/method.cpp)
target_link_libraries(serial_cpp PRIVATE OpenMP::OpenMP_CXX)

# OpenMP
add_executable(omp src/omp.cpp lib/ppm.cpp)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX)

# CUDA
add_executable(cuda src/cuda.cu lib/ppm.cpp)
target_link_libraries(cuda PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
