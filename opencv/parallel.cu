#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <filesystem>
#include <iostream>
#include <utility>
#include <string>
#include <map>
#include <cuda_runtime.h>

// Kernel CUDA para dithering com wavefront parallelism
__global__ void ditherKernel(float* input, unsigned char* output, int width, int height, 
                            const int2* offsets, const float* weights, int numWeights, 
                            int diagonal, int channels, int channel) {
    // Cada thread processa um pixel na diagonal atual
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = diagonal - y;
    
    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;
        float val = input[idx];
        
        // Aplicar threshold
        int new_val = (val < 128.0f) ? 0 : 255;
        
        // Escrever resultado
        if (channels == 1) {
            output[idx] = static_cast<unsigned char>(new_val);
        } else {
            // Para imagens coloridas, precisamos calcular o índice correto
            int color_idx = y * width * channels + x * channels + channel;
            output[color_idx] = static_cast<unsigned char>(new_val);
        }
        
        float error = val - new_val;
        
        // Difundir o erro para os vizinhos
        for (int i = 0; i < numWeights; i++) {
            int dx = offsets[i].x; // CORRIGIDO: era dy = offsets[i].x
            int dy = offsets[i].y; // CORRIGIDO: era dx = offsets[i].y
            int nx = x + dx;
            int ny = y + dy;
            
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                int neighbor_idx = ny * width + nx;
                atomicAdd(&input[neighbor_idx], error * weights[i]);
            }
        }
    }
}

std::map<std::pair<int, int>, float> create_dither_map(const std::string& modo) {
    std::map<std::pair<int, int>, float> dither_map;
    
    if (modo == "FloydSteinberg") {
        dither_map = {
            {{ 1,  0}, 7.f / 16.f},
            {{-1,  1}, 3.f / 16.f},
            {{ 0,  1}, 5.f / 16.f},
            {{ 1,  1}, 1.f / 16.f}
        };
    } else if (modo == "StevensonArce") {
        dither_map = {
            {{2,  0}, 32.f / 200.f},
            {{3,  1}, 12.f / 200.f},
            {{1,  1}, 26.f / 200.f},
            {{1,  1}, 30.f / 200.f},
            {{3,  1}, 16.f / 200.f},
            {{2,  2}, 12.f / 200.f},
            {{0,  2}, 26.f / 200.f},
            {{2,  2}, 12.f / 200.f},
            {{3,  3},  5.f / 200.f},
            {{1,  3}, 12.f / 200.f},
            {{1,  3}, 12.f / 200.f},
            {{3,  3},  5.f / 200.f}
        };
    } else if (modo == "Burkes") {
        dither_map = {
            {{ 1,  0}, 8.f / 32.f},
            {{ 2,  0}, 4.f / 32.f},
            {{-2,  1}, 2.f / 32.f},
            {{-1,  1}, 4.f / 32.f},
            {{ 0,  1}, 8.f / 32.f},
            {{ 1,  1}, 4.f / 32.f},
            {{ 2,  1}, 2.f / 32.f}
        };
    } else if (modo == "Sierra") {
        dither_map = {
            {{ 1,  0}, 5.f / 32.f},
            {{ 2,  0}, 3.f / 32.f},
            {{-2,  1}, 2.f / 32.f},
            {{-1,  1}, 4.f / 32.f},
            {{ 0,  1}, 5.f / 32.f},
            {{ 1,  1}, 4.f / 32.f},
            {{ 2,  1}, 2.f / 32.f},
            {{-1,  2}, 2.f / 32.f},
            {{ 0,  2}, 3.f / 32.f},
            {{ 1,  2}, 2.f / 32.f}
        };
    } else if (modo == "Stucki") {
        dither_map = {
            {{ 1,  0}, 8.f / 42.f},
            {{ 2,  0}, 4.f / 42.f},
            {{-2,  1}, 2.f / 42.f},
            {{-1,  1}, 4.f / 42.f},
            {{ 0,  1}, 8.f / 42.f},
            {{ 1,  1}, 4.f / 42.f},
            {{ 2,  1}, 2.f / 42.f},
            {{-2,  2}, 1.f / 42.f},
            {{-1,  2}, 2.f / 42.f},
            {{ 0,  2}, 4.f / 42.f},
            {{ 1,  2}, 2.f / 42.f},
            {{ 2,  2}, 1.f / 42.f}
        };
    } else if (modo == "JarvisJudiceNinke") {
        dither_map = {
            {{ 1,  0}, 7.f / 48.f},
            {{ 2,  0}, 5.f / 48.f},
            {{-2,  1}, 3.f / 48.f},
            {{-1,  1}, 5.f / 48.f},
            {{ 0,  1}, 7.f / 48.f},
            {{ 1,  1}, 5.f / 48.f},
            {{ 2,  1}, 3.f / 48.f},
            {{-2,  2}, 1.f / 48.f},
            {{-1,  2}, 3.f / 48.f},
            {{ 0,  2}, 5.f / 48.f},
            {{ 1,  2}, 3.f / 48.f},
            {{ 2,  2}, 1.f / 48.f}
        };
    }

    return dither_map;
}

void print_usage(const std::string& program_name, const std::unordered_set<std::string>& modos_validos) {
    std::cout << "Uso: " << program_name << " -i <imagem> -m <modo> [-g]\n";
    std::cout << "  -i <imagem>    Caminho para a imagem de entrada\n";
    std::cout << "  -m <modo>      Modo de dithering a ser utilizado\n";
    std::cout << "  -g             (Opcional) Ativa o modo escala de cinza\n";
    std::cout << "\nModos válidos:\n";
    for (const auto& modo : modos_validos)
        std::cout << "  - " << modo << "\n";
    std::cout << "\nExemplo:\n";
    std::cout << "  " << program_name << " -i ../img/city.png -m FloydSteinberg -g\n";
}

void DitheringDifusionErrorDiagonalCUDA(const cv::Mat& inputImage, cv::Mat& outputImage, 
                                      const std::map<std::pair<int, int>, float>& metodo) {
    int height = inputImage.rows;
    int width = inputImage.cols;
    int channels = inputImage.channels();

    if (channels != 1 && channels != 3) {
        std::cerr << "Número de canais não suportado: " << channels << std::endl;
        return;
    }

    // Converter para float
    cv::Mat imagemFloat;
    inputImage.convertTo(imagemFloat, (channels == 1 ? CV_32FC1 : CV_32FC3));
    outputImage = cv::Mat::zeros(inputImage.size(), (channels == 1 ? CV_8UC1 : CV_8UC3));

    // Preparar dados para CUDA
    int numWeights = metodo.size();
    int2* offsets = new int2[numWeights];
    float* weights = new float[numWeights];
    
    int i = 0;
    for (const auto& [offset, weight] : metodo) {
        offsets[i] = {offset.first, offset.second};
        weights[i] = weight;
        i++;
    }

    // Alocar memória na GPU
    int2* d_offsets;
    float* d_weights;
    float* d_input;
    unsigned char* d_output;
    
    size_t inputSize = height * width * channels * sizeof(float);
    size_t outputSize = height * width * channels * sizeof(unsigned char);
    
    cudaMalloc(&d_offsets, numWeights * sizeof(int2));
    cudaMalloc(&d_weights, numWeights * sizeof(float));
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
    
    // Copiar offsets e weights para a GPU
    cudaMemcpy(d_offsets, offsets, numWeights * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, numWeights * sizeof(float), cudaMemcpyHostToDevice);

    // Processar cada canal separadamente
    for (int channel = 0; channel < channels; ++channel) {
        // Copiar dados do canal para a GPU
        if (channels == 1) {
            cudaMemcpy(d_input, imagemFloat.ptr<float>(), height * width * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            // Para imagens coloridas, precisamos copiar apenas o canal atual
            std::vector<float> channelData(height * width);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    channelData[y * width + x] = imagemFloat.at<cv::Vec3f>(y, x)[channel];
                }
            }
            cudaMemcpy(d_input, channelData.data(), height * width * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        // Limpar output na GPU
        cudaMemset(d_output, 0, outputSize);
        
        // Configurar kernel
        dim3 blockDim(32); // 32 threads por bloco
        dim3 gridDim((height + blockDim.x - 1) / blockDim.x); // Número de blocos necessário
        
        // Processar cada diagonal
        for (int diagonal = 0; diagonal <= width + height - 2; ++diagonal) {
            ditherKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, 
                                              d_offsets, d_weights, numWeights, 
                                              diagonal, channels, channel); // CORRIGIDO: d_output + channel -> d_output
            cudaDeviceSynchronize(); // Sincronizar após cada diagonal
        }
        
        // Copiar resultados de volta para a CPU
        if (channels == 1) {
            cudaMemcpy(outputImage.ptr(), d_output, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        } else {
            std::vector<unsigned char> channelData(height * width);
            cudaMemcpy(channelData.data(), d_output, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost); // CORRIGIDO: d_output + channel -> d_output
            
            // Copiar de volta para a imagem de saída
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    outputImage.at<cv::Vec3b>(y, x)[channel] = channelData[y * width + x];
                }
            }
        }
    }
    
    // Liberar memória da GPU
    cudaFree(d_offsets);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    
    delete[] offsets;
    delete[] weights;
}

int main(int argc, char** argv) {
    std::string img_path;
    std::string modo;
    std::unordered_set<std::string> modos_validos = {"FloydSteinberg", "StevensonArce", "Burkes", "Sierra", "Stucki", "JarvisJudiceNinke"};
    bool grayscale = false;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];

        if (flag == "-i" && i + 1 < argc) {
            img_path = argv[++i];
        } else if (flag == "-m" && i + 1 < argc) {
            modo = argv[++i];
        } else if (flag == "-g") {
            grayscale = true;
        }
    }

    if (img_path.empty() || modo.empty() || modos_validos.find(modo) == modos_validos.end()) {
        print_usage(argv[0], modos_validos);
        return 1;
    }

    std::filesystem::path p(img_path);
    std::string img_name = p.stem().string();

    if (!std::filesystem::exists("../out/" + img_name))
        std::filesystem::create_directories("../out/" + img_name);

    std::cout << "Processando imagem " << img_path << " pelo método " << modo << "\n";

    std::map<std::pair<int, int>, float> dither_map = create_dither_map(modo);
    
    cv::Mat img;
    cv::Mat resultado;
    img = grayscale ? cv::imread(img_path, cv::IMREAD_GRAYSCALE): cv::imread(img_path, cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "Erro ao carregar a imagem! Verifique o caminho passado." << std::endl;
        return 1;
    }

    DitheringDifusionErrorDiagonalCUDA(img, resultado, dither_map);
    
    std::string label = grayscale ? modo + " Error Difusion (Grayscale) for " + img_name  : modo + " Error Difusion for " + img_name;
    std::string nome_arquivo = grayscale ? modo + "Grayscale": modo;
    std::string output_path = "../out/" + img_name + "/" + nome_arquivo + ".png";
    
    cv::imwrite(output_path, resultado);

    cv::imshow(label, resultado);

    // Loop: espera por tecla ou fechamento da janela
    while (cv::waitKey(30) < 0 && cv::getWindowProperty(label, cv::WND_PROP_VISIBLE) >= 1);

    std::cout << "Imagem processada e salva como: " << output_path << std::endl;

    return 0;
}