#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstring>
#include "method.hpp"
#include "ppm.hpp"

#define BLOCK_SIZE 32
#define DEFAULT_P 0.5f

__device__ void apply_method(float* image, int x, int y, int width, int height,
                            const int* dx, const int* dy, const float* coef, int n) {
    int idx = y * width + x;
    float old_pixel = image[idx];
    float new_pixel = old_pixel < 0.5f ? 0.0f : 1.0f;
    float quant_error = old_pixel - new_pixel;
    image[idx] = new_pixel;

    for (int i = 0; i < n; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            atomicAdd(&image[ny * width + nx], quant_error * coef[i]);
        }
    }
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

__global__ void process_diagonal(float* image, int diag, int width, int height,
                                 const int* dx, const int* dy, const float* coef, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x, y;
    get_coordinates_from_diagonal(idx, diag, width, height, &x, &y);
    if (x >= 0 && y >= 0 && x < width && y < height) {
        apply_method(image, x, y, width, height, dx, dy, coef, n);
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

void prepare_method_arrays(const std::string& method_name, int* dx, int* dy, float* coef, int& n) {
    const auto& method = get_method(method_name);
    n = 0;
    for (const auto& kv : method) {
        dx[n] = kv.first.first;
        dy[n] = kv.first.second;
        coef[n] = kv.second;
        n++;
    }
}

void process_channel(float* deviceChannel, int width, int height, const std::string& method_name) {
    int dx[16], dy[16];
    float coef[16];
    int n;
    prepare_method_arrays(method_name, dx, dy, coef, n);

    int *d_dx, *d_dy;
    float *d_coef;
    cudaMalloc(&d_dx, n * sizeof(int));
    cudaMalloc(&d_dy, n * sizeof(int));
    cudaMalloc(&d_coef, n * sizeof(float));
    cudaMemcpy(d_dx, dx, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coef, coef, n * sizeof(float), cudaMemcpyHostToDevice);

    double start = omp_get_wtime();

    int maxDiagonals = width + height - 1;
    for (int diag = 0; diag < maxDiagonals; ++diag) {
        int numPixels = count_pixels_in_diagonal(diag, width, height);
        int threadsPerBlock = 32;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
        process_diagonal<<<blocks, threadsPerBlock>>>(deviceChannel, diag, width, height, d_dx, d_dy, d_coef, n);
        cudaDeviceSynchronize();
    }

    double end = omp_get_wtime();
    std::cout << "Execution time: " << (end - start) * 1000 << " ms" << std::endl;

    cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_coef);
}

__global__ void stochastic_method_kernel(
    float* image, int width, int height, float p,
    const int* dx, const int* dy, const float* coef, int n) 
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float old_pixel = image[idx];
    float new_pixel = old_pixel < 0.5f ? 0.0f : 1.0f;
    float error = old_pixel - new_pixel;
    image[idx] = new_pixel;

    curandState state;
    curand_init(1337 + idx, 0, 0, &state);

    for (int i = 0; i < n; ++i) {
        float noise = (curand_uniform(&state) - 0.5f) * 2.0f; // [-1, 1]
        float noisy_coef = coef[i] + p * noise * coef[i];
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            atomicAdd(&image[ny * width + nx], error * noisy_coef);
        }
    }
}

void process_stochastic(float* d_image, int width, int height, float p, const std::string& method_name) {
    int dx[16], dy[16];
    float coef[16];
    int n;
    prepare_method_arrays(method_name, dx, dy, coef, n);

    int *d_dx, *d_dy;
    float *d_coef;
    cudaMalloc(&d_dx, n * sizeof(int));
    cudaMalloc(&d_dy, n * sizeof(int));
    cudaMalloc(&d_coef, n * sizeof(float));
    cudaMemcpy(d_dx, dx, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coef, coef, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    double start = omp_get_wtime();

    stochastic_method_kernel<<<blocks, threads>>>(d_image, width, height, p, d_dx, d_dy, d_coef, n);
    cudaDeviceSynchronize();

    double end = omp_get_wtime();
    std::cout << "Execution time: " << (end - start) * 1000 << " ms" << std::endl;

    cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_coef);
}

int main(int argc, char* argv[]) {
        if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> <output.ppm> <method> [p=0.5] [stochastic=1] [-g]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  method:     Error diffusion method (FloydSteinberg, StevensonArce, Burkes, Sierra, Stucki, JarvisJudiceNinke)" << std::endl;
        std::cerr << "  p:          Noise parameter for stochastic dithering (default: 0.5)" << std::endl;
        std::cerr << "  stochastic: Use stochastic dithering (1) or standard (0) (default: 1)" << std::endl;
        std::cerr << "  -g:         Convert to grayscale before processing" << std::endl;
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    std::string method_name = argv[3];
    float p = DEFAULT_P;
    bool stochastic = true;
    bool grayscale = false;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0) {
            grayscale = true;
        } else if (i == 4 && argv[i][0] != '-') {
            p = atof(argv[i]);
        } else if (i == 5 && argv[i][0] != '-') {
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

        if (stochastic) {
            process_stochastic(d_gray, width, height, p, method_name);
        } else {
            process_channel(d_gray, width, height, method_name);
        }

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

        if (stochastic) {
            process_stochastic(dR, width, height, p, method_name);
            process_stochastic(dG, width, height, p, method_name);
            process_stochastic(dB, width, height, p, method_name);
        } else {
            process_channel(dR, width, height, method_name);
            process_channel(dG, width, height, method_name);
            process_channel(dB, width, height, method_name);
        }

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