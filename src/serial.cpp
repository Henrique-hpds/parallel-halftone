#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>

bool loadPGM(const std::string& filename, std::vector<unsigned char>& image, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::string line;
    std::getline(file, line);
    if (line != "P5") return false;

    // Pula comentários
    do {
        std::getline(file, line);
    } while (line[0] == '#');

    std::istringstream iss(line);
    iss >> width >> height;

    int maxVal;
    file >> maxVal;
    file.get();

    image.resize(width * height);
    file.read(reinterpret_cast<char*>(image.data()), width * height);

    return true;
}

void savePGM(const std::string& filename, const std::vector<unsigned char>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), width * height);
}

int count_pixels_in_diagonal(int diag, int width, int height) {
    int start_x = std::max(0, diag - height + 1);
    int end_x = std::min(width - 1, diag);
    return end_x - start_x + 1;
}

void process_diagonal_serial(std::vector<float>& image, int diag, int width, int height) {
    int start_x = std::max(0, diag - height + 1);
    int end_x = std::min(width - 1, diag);

    for (int x = start_x; x <= end_x; x++) {
        int y = diag - x;
        if (y < 0 || y >= height) continue;

        int idx = y * width + x;
        float old_pixel = image[idx];
        float new_pixel = old_pixel < 0.5f ? 0.0f : 1.0f;
        float quant_error = old_pixel - new_pixel;
        image[idx] = new_pixel;

        // Propagação de erro (Floyd-Steinberg)
        if (x + 1 < width)
            image[y * width + (x + 1)] += quant_error * 7.0f / 16.0f;
        if (x - 1 >= 0 && y + 1 < height)
            image[(y + 1) * width + (x - 1)] += quant_error * 3.0f / 16.0f;
        if (y + 1 < height)
            image[(y + 1) * width + x] += quant_error * 5.0f / 16.0f;
        if (x + 1 < width && y + 1 < height)
            image[(y + 1) * width + (x + 1)] += quant_error * 1.0f / 16.0f;
    }
}

int main() {
    std::vector<unsigned char> hostImage;
    int width, height;

    if (!loadPGM("../sample_640_426.pgm", hostImage, width, height)) {
        std::cerr << "Falha ao carregar a imagem." << std::endl;
        return -1;
    }

    const int imageSize = width * height;
    std::vector<float> imageFloat(imageSize);

    // Normalizar para 0.0 a 1.0
    for (int i = 0; i < imageSize; i++) {
        imageFloat[i] = hostImage[i] / 255.0f;
    }

    int maxDiagonals = width + height - 1;

    double omp_start = omp_get_wtime();    
    
    // Processar cada diagonal (serial, uma a uma)
    for (int diag = 0; diag < maxDiagonals; diag++) {
        process_diagonal_serial(imageFloat, diag, width, height);
    }

    double omp_end = omp_get_wtime();
    std::cout << "Tempo de execução dos kernels (OpenMP): " << (omp_end - omp_start) * 1000 << " ms" << std::endl;

    // Converter de volta para binário (0 ou 255)
    for (int i = 0; i < imageSize; i++) {
        float clamped = fminf(fmaxf(imageFloat[i], 0.0f), 1.0f);
        hostImage[i] = (clamped < 0.5f) ? 0 : 255;
    }

    savePGM("../output_serial.pgm", hostImage, width, height);
    std::cout << "Imagem salva como output_serial.pgm" << std::endl;

    return 0;
}