cmake_minimum_required(VERSION 3.18)
project(FloydSteinbergHalftone LANGUAGES CXX CUDA)

# Configuração geral
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

# Executável Serial (compila apenas com C++)
add_executable(floydsteinberg_serial serial.cpp)
target_link_libraries(floydsteinberg_serial PRIVATE OpenMP::OpenMP_CXX)

# Executável CUDA Paralelo
add_executable(floydsteinberg_parallel cu.cu)
target_link_libraries(floydsteinberg_parallel PRIVATE OpenMP::OpenMP_CXX)