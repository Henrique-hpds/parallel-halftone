#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <omp.h>
#include "ppm.hpp"

#define DEFAULT_P 0.5f

// Standard Floyd-Steinberg dithering
void apply_floyd_steinberg_omp(float* image, int width, int height) {
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float old_pixel = image[idx];
            float new_pixel = old_pixel < 0.5f ? 0.0f : 1.0f;
            float quant_error = old_pixel - new_pixel;
            image[idx] = new_pixel;

            if (x + 1 < width)
                #pragma omp atomic
                image[y * width + (x + 1)] += quant_error * 7.0f / 16.0f;
            if (x - 1 >= 0 && y + 1 < height)
                #pragma omp atomic
                image[(y + 1) * width + (x - 1)] += quant_error * 3.0f / 16.0f;
            if (y + 1 < height)
                #pragma omp atomic
                image[(y + 1) * width + x] += quant_error * 5.0f / 16.0f;
            if (x + 1 < width && y + 1 < height)
                #pragma omp atomic
                image[(y + 1) * width + (x + 1)] += quant_error * 1.0f / 16.0f;
        }
    }
}

// Stochastic Floyd-Steinberg dithering
void apply_floyd_steinberg_stochastic_omp(float* image, int width, int height, float p) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        #pragma omp for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float old_pixel = image[idx];
                float new_pixel = old_pixel < 0.5f ? 0.0f : 1.0f;
                float error = old_pixel - new_pixel;
                image[idx] = new_pixel;

                // Base error weights
                float e1 = 7.0f / 16.0f, e2 = 5.0f / 16.0f, e3 = 3.0f / 16.0f, e4 = 1.0f / 16.0f;

                // Add random noise to the weights
                float r1 = dis(gen) * (5.0f / 16.0f);
                float r2 = dis(gen) * (1.0f / 16.0f);
                e1 += p * r1;
                e2 -= p * r1;
                e3 += p * r2;
                e4 -= p * r2;

                if (x + 1 < width)
                    #pragma omp atomic
                    image[y * width + (x + 1)] += error * e1;
                if (x - 1 >= 0 && y + 1 < height)
                    #pragma omp atomic
                    image[(y + 1) * width + (x - 1)] += error * e3;
                if (y + 1 < height)
                    #pragma omp atomic
                    image[(y + 1) * width + x] += error * e2;
                if (x + 1 < width && y + 1 < height)
                    #pragma omp atomic
                    image[(y + 1) * width + (x + 1)] += error * e4;
            }
        }
    }
}

void process_channel_omp(float* channel, int width, int height, bool stochastic, float p) {
    if (stochastic) {
        apply_floyd_steinberg_stochastic_omp(channel, width, height, p);
    } else {
        apply_floyd_steinberg_omp(channel, width, height);
    }
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
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            PPMPixel& p = img->data[i];
            gray[i] = (0.299f * p.red + 0.587f * p.green + 0.114f * p.blue) / 255.0f;
        }

        double start = omp_get_wtime();

        process_channel_omp(gray.data(), width, height, stochastic, p);

        double end = omp_get_wtime();
        std::cout << "OpenMP execution time (" << omp_get_max_threads() << " threads): " << (end - start) * 1000 << " ms" << std::endl;

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            unsigned char val = (gray[i] < 0.35f ? 0 : 255);
            img->data[i].red = val;
            img->data[i].green = val;
            img->data[i].blue = val;
        }
    } else {
        std::vector<float> R(size), G(size), B(size);
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            R[i] = img->data[i].red / 255.0f;
            G[i] = img->data[i].green / 255.0f;
            B[i] = img->data[i].blue / 255.0f;
        }

        double start = omp_get_wtime();

        process_channel_omp(R.data(), width, height, stochastic, p);
        process_channel_omp(G.data(), width, height, stochastic, p);
        process_channel_omp(B.data(), width, height, stochastic, p);

        double end = omp_get_wtime();
        std::cout << "OpenMP execution time (" << omp_get_max_threads() << " threads): " << (end - start) * 1000 << " ms" << std::endl;

        #pragma omp parallel for
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