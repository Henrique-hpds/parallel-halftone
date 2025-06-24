#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstring>
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " [-g] <arquivo.ppm>" << std::endl;
        return 1;
    }

    bool grayscale = false;
    const char* filename = nullptr;

    if (strcmp(argv[1], "-g") == 0) {
        grayscale = true;
        if (argc < 3) {
            std::cerr << "Faltando caminho para imagem PPM." << std::endl;
            return 1;
        }
        filename = argv[2];
    } else {
        filename = argv[1];
    }

    PPMImage* colorImage = readPPM(filename);
    if (!colorImage) {
        std::cerr << "Erro ao carregar imagem PPM" << std::endl;
        return -1;
    }

    int width = colorImage->x;
    int height = colorImage->y;
    const int imageSize = width * height;

    double start = omp_get_wtime();

    if (grayscale) {
        std::vector<float> hostGray(imageSize);
        for (int i = 0; i < imageSize; ++i) {
            PPMPixel& p = colorImage->data[i];
            hostGray[i] = (0.299f * p.red + 0.587f * p.green + 0.114f * p.blue) / 255.0f;
        }

        float* deviceGray;
        cudaMalloc(&deviceGray, imageSize * sizeof(float));
        cudaMemcpy(deviceGray, hostGray.data(), imageSize * sizeof(float), cudaMemcpyHostToDevice);

        process_channel(deviceGray, width, height);

        cudaMemcpy(hostGray.data(), deviceGray, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(deviceGray);

        for (int i = 0; i < imageSize; ++i) {
            unsigned char val = (hostGray[i] < 0.35f ? 0 : 255);
            colorImage->data[i].red = val;
            colorImage->data[i].green = val;
            colorImage->data[i].blue = val;
        }

    } else {
        // Colorido: processar R, G, B separadamente
        std::vector<float> hostR(imageSize), hostG(imageSize), hostB(imageSize);
        for (int i = 0; i < imageSize; ++i) {
            hostR[i] = colorImage->data[i].red / 255.0f;
            hostG[i] = colorImage->data[i].green / 255.0f;
            hostB[i] = colorImage->data[i].blue / 255.0f;
        }

        float *deviceR, *deviceG, *deviceB;
        cudaMalloc(&deviceR, imageSize * sizeof(float));
        cudaMalloc(&deviceG, imageSize * sizeof(float));
        cudaMalloc(&deviceB, imageSize * sizeof(float));

        cudaMemcpy(deviceR, hostR.data(), imageSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceG, hostG.data(), imageSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB.data(), imageSize * sizeof(float), cudaMemcpyHostToDevice);

        process_channel(deviceR, width, height);
        process_channel(deviceG, width, height);
        process_channel(deviceB, width, height);

        cudaMemcpy(hostR.data(), deviceR, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostG.data(), deviceG, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostB.data(), deviceB, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(deviceR);
        cudaFree(deviceG);
        cudaFree(deviceB);

        for (int i = 0; i < imageSize; ++i) {
            colorImage->data[i].red   = (hostR[i] < 0.35f ? 0 : 255);
            colorImage->data[i].green = (hostG[i] < 0.35f ? 0 : 255);
            colorImage->data[i].blue  = (hostB[i] < 0.35f ? 0 : 255);
        }
    }

    double end = omp_get_wtime();
    std::cout << "Tempo de execução CUDA: " << (end - start) * 1000 << " ms" << std::endl;

    writePPM(colorImage, "../out/output_cuda.ppm");
    std::cout << "Imagem salva como ../out/output_cuda.ppm" << std::endl;

    free(colorImage->data);
    free(colorImage);
    return 0;
}
