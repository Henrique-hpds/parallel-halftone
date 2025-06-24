#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <omp.h>

#define WIDTH 8
#define HEIGHT 8

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
}

int count_pixels_in_diagonal(int diag, int width, int height) {
    int count = 0;
    for (int i = 0; i <= diag; i++) {
        int x = i;
        int y = diag - i;
        if (x < width && y < height) {
            count++;
        }
    }
    return count;
}

void savePGM(const std::string& filename, const std::vector<unsigned char>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Erro ao abrir o arquivo para escrita: " << filename << std::endl;
        return;
    }

    // Escreve o cabeçalho PGM
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), width * height);
    file.close();
}

bool loadPGM(const std::string& filename, std::vector<unsigned char>& image, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Erro ao abrir o arquivo: " << filename << std::endl;
        return false;
    }

    std::string line;
    // Ler mágica
    std::getline(file, line);
    if (line != "P5") {
        std::cerr << "Formato inválido (não é P5)" << std::endl;
        return false;
    }

    // Ignorar comentários
    do {
        std::getline(file, line);
    } while (line[0] == '#');

    // Ler dimensões
    std::istringstream iss(line);
    iss >> width >> height;

    // Ler valor máximo (deve ser 255)
    int maxVal;
    file >> maxVal;
    file.get(); // Consumir o '\n'

    if (maxVal != 255) {
        std::cerr << "Só suportado PGM com maxVal 255" << std::endl;
        return false;
    }

    image.resize(width * height);
    file.read(reinterpret_cast<char*>(image.data()), width * height);
    return true;
}

int main() {
    std::vector<unsigned char> hostImage;
    int width, height;

    // Carregar imagem PGM de entrada
    if (!loadPGM("../sample_640_426.pgm", hostImage, width, height)) {
        return -1;
    }

    const int imageSize = width * height;
    std::vector<float> hostImageFloat(imageSize);

    // Normalizar para 0.0 a 1.0 antes de enviar para GPU
    for (int i = 0; i < imageSize; i++) {
        hostImageFloat[i] = hostImage[i] / 255.0f;
    }

    // Alocar memória no device
    float* deviceImage;
    cudaMalloc(&deviceImage, imageSize * sizeof(float));
    cudaMemcpy(deviceImage, hostImageFloat.data(), imageSize * sizeof(float), cudaMemcpyHostToDevice);

    int maxDiagonals = width + height - 1;

    double omp_start = omp_get_wtime();    

    // Processar cada diagonal (wavefront)
    for (int diag = 0; diag < maxDiagonals; diag++) {
        int numPixels = count_pixels_in_diagonal(diag, width, height);
        int threadsPerBlock = 32;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

        process_diagonal<<<blocks, threadsPerBlock>>>(deviceImage, diag, width, height);
        cudaDeviceSynchronize();
    }

    double omp_end = omp_get_wtime();
    std::cout << "Tempo de execução dos kernels (OpenMP): " << (omp_end - omp_start) * 1000 << " ms" << std::endl;

    // Copiar o resultado de volta
    cudaMemcpy(hostImageFloat.data(), deviceImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < imageSize; i++) {
        float clamped = fminf(fmaxf(hostImageFloat[i], 0.0f), 1.0f);
        hostImage[i] = (clamped < 0.35f) ? 0 : 255;  // Força saída binária (preto ou branco)
    }

    // Salvar imagem final
    savePGM("../output_cuda.pgm", hostImage, width, height);
    std::cout << "Imagem salva como output.pgm" << std::endl;

    // Limpeza
    cudaFree(deviceImage);
    return 0;
}