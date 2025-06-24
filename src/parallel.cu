#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include "ppm.hpp"

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

int main() {
    PPMImage* colorImage = readPPM("../img/vazio_roxo.ppm");
    if (!colorImage) {
        std::cerr << "Erro ao carregar imagem PPM" << std::endl;
        return -1;
    }

    int width = colorImage->x;
    int height = colorImage->y;
    const int imageSize = width * height;

    std::vector<float> hostImageFloat(imageSize);

    // Converter para escala de cinza
    for (int i = 0; i < imageSize; ++i) {
        PPMPixel& p = colorImage->data[i];
        hostImageFloat[i] = (0.299f * p.red + 0.587f * p.green + 0.114f * p.blue) / 255.0f;
    }

    // CUDA malloc + cópia
    float* deviceImage;
    cudaMalloc(&deviceImage, imageSize * sizeof(float));
    cudaMemcpy(deviceImage, hostImageFloat.data(), imageSize * sizeof(float), cudaMemcpyHostToDevice);

    double start = omp_get_wtime();

    int maxDiagonals = width + height - 1;
    for (int diag = 0; diag < maxDiagonals; ++diag) {
        int numPixels = count_pixels_in_diagonal(diag, width, height);
        int threadsPerBlock = 32;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

        process_diagonal<<<blocks, threadsPerBlock>>>(deviceImage, diag, width, height);
        cudaDeviceSynchronize();
    }

    double end = omp_get_wtime();
    std::cout << "Tempo de execução CUDA: " << (end - start) * 1000 << " ms" << std::endl;

    cudaMemcpy(hostImageFloat.data(), deviceImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Binarizar e salvar no formato PPM
    for (int i = 0; i < imageSize; ++i) {
        float clamped = fminf(fmaxf(hostImageFloat[i], 0.0f), 1.0f);
        unsigned char bin = (clamped < 0.35f) ? 0 : 255;
        colorImage->data[i].red = bin;
        colorImage->data[i].green = bin;
        colorImage->data[i].blue = bin;
    }

    writePPM(colorImage, "../out/output_cuda.ppm");
    std::cout << "Imagem salva como output_cuda.ppm" << std::endl;

    cudaFree(deviceImage);
    free(colorImage->data);
    free(colorImage);
    return 0;
}
