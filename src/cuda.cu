#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstring>
#include "ppm.hpp"

#define BLOCK_SIZE 32
#define DEFAULT_P 0.5f

// Diagonal processing functions for standard version
__device__ void apply_floyd_steinberg(float* image, int x, int y, int width, int height) {
    int idx = y * width + x;
    float old_pixel = image[idx];
    float new_pixel = old_pixel < 0.5f ? 0.0f : 1.0f;
    float quant_error = old_pixel - new_pixel;
    image[idx] = new_pixel;

    if (x + 1 < width)
        atomicAdd(&image[y * width + (x + 1)], quant_error * 7.0f / 16.0f);
    if (x - 1 >= 0 && y + 1 < height)
        atomicAdd(&image[(y + 1) * width + (x - 1)], quant_error * 3.0f / 16.0f);
    if (y + 1 < height)
        atomicAdd(&image[(y + 1) * width + x], quant_error * 5.0f / 16.0f);
    if (x + 1 < width && y + 1 < height)
        atomicAdd(&image[(y + 1) * width + (x + 1)], quant_error * 1.0f / 16.0f);
}

__device__ void get_coordinates_from_diagonal(int idx_in_diag, int diag, int width, int height, int* x, int* y) {
    int count = 0;
    for (int i = 0; i <= diag; i++) {
        int px = i;
        int py = diag - i;
        if (px < width && py < height) {
            if (count == idx_in_diag) {
                *x = px;
                *y = py;
                return;
            }
            count++;
        }
    }
    *x = -1;
    *y = -1;
}

__global__ void process_diagonal(float* image, int diag, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x, y;
    get_coordinates_from_diagonal(idx, diag, width, height, &x, &y);
    if (x >= 0 && y >= 0 && x < width && y < height) {
        apply_floyd_steinberg(image, x, y, width, height);
    }
}

int count_pixels_in_diagonal(int diag, int width, int height) {
    int count = 0;
    for (int i = 0; i <= diag; i++) {
        int x = i;
        int y = diag - i;
        if (x < width && y < height) count++;
    }
    return count;
}

void process_channel(float* deviceChannel, int width, int height) {
    int maxDiagonals = width + height - 1;
    for (int diag = 0; diag < maxDiagonals; ++diag) {
        int numPixels = count_pixels_in_diagonal(diag, width, height);
        int threadsPerBlock = 32;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
        process_diagonal<<<blocks, threadsPerBlock>>>(deviceChannel, diag, width, height);
        cudaDeviceSynchronize();
    }
}

// Stochastic version functions
__global__ void floyd_steinberg_kernel(
    float* image, int width, int height,
    float p) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float old_pixel = image[idx];
    float new_pixel = old_pixel < 0.5f ? 0.0f : 1.0f;
    float error = old_pixel - new_pixel;
    image[idx] = new_pixel;

    float e1 = 7.0f / 16.0f, e2 = 5.0f / 16.0f, e3 = 3.0f / 16.0f, e4 = 1.0f / 16.0f;

    curandState state;
    curand_init(1337 + idx, 0, 0, &state);
    float r1 = curand_uniform(&state) * (10.0f / 16.0f) - (5.0f / 16.0f);
    float r2 = curand_uniform(&state) * (2.0f / 16.0f) - (1.0f / 16.0f);
    e1 += p * r1;
    e2 -= p * r1;
    e3 += p * r2;
    e4 -= p * r2;

    if (x + 1 < width)
        atomicAdd(&image[y * width + (x + 1)], error * e1);
    if (x - 1 >= 0 && y + 1 < height)
        atomicAdd(&image[(y + 1) * width + (x - 1)], error * e3);
    if (y + 1 < height)
        atomicAdd(&image[(y + 1) * width + x], error * e2);
    if (x + 1 < width && y + 1 < height)
        atomicAdd(&image[(y + 1) * width + (x + 1)], error * e4);
}

void process_stochastic(float* d_image, int width, int height, float p = DEFAULT_P) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    double start = omp_get_wtime();
    floyd_steinberg_kernel<<<blocks, threads>>>(d_image, width, height, p);
    cudaDeviceSynchronize();
    double end = omp_get_wtime();

    std::cout << "CUDA execution time: " << (end - start) * 1000 << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> <output.ppm> [p=0.5] [stochastic=1] [-g]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  p:          Noise parameter for stochastic dithering (default: 0.5)" << std::endl;
        std::cerr << "  stochastic: Use stochastic dithering (1) or standard (0) (default: 1)" << std::endl;
        std::cerr << "  -g:         Convert to grayscale before processing" << std::endl;
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    float p = DEFAULT_P;
    bool stochastic = true;
    bool grayscale = false;

    // Parse optional arguments
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0) {
            grayscale = true;
        } else if (i == 3 && argv[i][0] != '-') {
            p = atof(argv[i]);
        } else if (i == 4 && argv[i][0] != '-') {
            stochastic = atoi(argv[i]) != 0;
        }
    }

    PPMImage* img = readPPM(input_filename);
    if (!img) {
        std::cerr << "Error loading image." << std::endl;
        return 1;
    }

    int width = img->x, height = img->y;
    int size = width * height;

    if (grayscale) {
        std::vector<float> gray(size);
        for (int i = 0; i < size; ++i) {
            PPMPixel& p = img->data[i];
            gray[i] = (0.299f * p.red + 0.587f * p.green + 0.114f * p.blue) / 255.0f;
        }

        float* d_gray;
        cudaMalloc(&d_gray, size * sizeof(float));
        cudaMemcpy(d_gray, gray.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        double start = omp_get_wtime();

        if (stochastic) {
            process_stochastic(d_gray, width, height, p);
        } else {
            process_channel(d_gray, width, height);
        }

        double end = omp_get_wtime();
        std::cout << "Execution time: " << (end - start) * 1000 << " ms" << std::endl;

        cudaMemcpy(gray.data(), d_gray, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_gray);

        for (int i = 0; i < size; ++i) {
            unsigned char val = (gray[i] < 0.35f ? 0 : 255);
            img->data[i].red = val;
            img->data[i].green = val;
            img->data[i].blue = val;
        }
    } else {
        std::vector<float> R(size), G(size), B(size);
        for (int i = 0; i < size; i++) {
            R[i] = img->data[i].red / 255.0f;
            G[i] = img->data[i].green / 255.0f;
            B[i] = img->data[i].blue / 255.0f;
        }

        float *dR, *dG, *dB;
        cudaMalloc(&dR, size * sizeof(float));
        cudaMalloc(&dG, size * sizeof(float));
        cudaMalloc(&dB, size * sizeof(float));

        cudaMemcpy(dR, R.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dG, G.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        double start = omp_get_wtime();

        if (stochastic) {
            process_stochastic(dR, width, height, p);
            process_stochastic(dG, width, height, p);
            process_stochastic(dB, width, height, p);
        } else {
            process_channel(dR, width, height);
            process_channel(dG, width, height);
            process_channel(dB, width, height);
        }

        double end = omp_get_wtime();
        std::cout << "Execution time: " << (end - start) * 1000 << " ms";
        std::cout << " | Mode: " << (stochastic ? "Stochastic" : "Standard") << std::endl;

        cudaMemcpy(R.data(), dR, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(G.data(), dG, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B.data(), dB, size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dR); cudaFree(dG); cudaFree(dB);

        for (int i = 0; i < size; i++) {
            img->data[i].red   = (R[i] < 0.35f ? 0 : 255);
            img->data[i].green = (G[i] < 0.35f ? 0 : 255);
            img->data[i].blue  = (B[i] < 0.35f ? 0 : 255);
        }
    }

    writePPM(img, output_filename);
    std::cout << "Image saved as " << output_filename << std::endl;

    free(img->data);
    free(img);
    return 0;
}