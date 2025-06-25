cmake_minimum_required(VERSION 3.18)
project(FloydSteinbergHalftone LANGUAGES CXX CUDA)

# Configuração geral
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Executável Serial (compila apenas com C++)
add_executable(serial src/serial.cpp lib/ppm.cpp)
target_link_libraries(serial PRIVATE OpenMP::OpenMP_CXX)

# Executável OpenMP Paralelo
add_executable(omp src/omp.cpp lib/ppm.cpp)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX)

# Executável CUDA Paralelo
add_executable(cuda src/cuda.cu lib/ppm.cpp)
target_link_libraries(cuda PRIVATE OpenMP::OpenMP_CXX)
