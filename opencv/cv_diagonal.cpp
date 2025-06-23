#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <filesystem>
#include <iostream>
#include <utility>
#include <string>
#include <map>

std::map<std::pair<int, int>, float> create_dither_map(const std::string& modo) {

    std::map<std::pair<int, int>, float> dither_map;
    
    if (modo == "FloydSteinberg") {
        dither_map = {
            {{ 1,  0}, 7.f / 16.f},
            {{-1, -1}, 3.f / 16.f},
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

void DitheringDifusionErrorDiagonal(const cv::Mat& inputImage, cv::Mat& outputImage, const std::map<std::pair<int, int>, float>& metodo) {
    int height = inputImage.rows;
    int width = inputImage.cols;
    int channels = inputImage.channels();

    if (channels != 1 && channels != 3) {
        std::cerr << "Número de canais não suportado: " << channels << std::endl;
        return;
    }

    cv::Mat imagemFloat;
    inputImage.convertTo(imagemFloat, (channels == 1 ? CV_32FC1 : CV_32FC3));
    outputImage = cv::Mat::zeros(inputImage.size(), (channels == 1 ? CV_8UC1 : CV_8UC3));

    for (int channel = 0; channel < channels; ++channel) {
        cv::Mat f(height, width, CV_32FC1);

        // Copiar canal
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                f.at<float>(y, x) = (channels == 1)
                    ? imagemFloat.at<float>(y, x)
                    : imagemFloat.at<cv::Vec3f>(y, x)[channel];

        // Total de diagonais = (width + height - 1)
        for (int diag = 0; diag <= width + height - 2; ++diag) {
            for (int y = 0; y < height; ++y) {
                int x = diag - y;
                if (x >= 0 && x < width) {
                    float val = f.at<float>(y, x);
                    int novo_val = (val < 128.0f) ? 0 : 255;

                    if (channels == 1)
                        outputImage.at<uchar>(y, x) = static_cast<uchar>(novo_val);
                    else
                        outputImage.at<cv::Vec3b>(y, x)[channel] = static_cast<uchar>(novo_val);

                    float erro = val - novo_val;

                    for (const auto& [offset, peso] : metodo) {
                        int dy = offset.first;
                        int dx = offset.second;
                        int ny = y + dy;
                        int nx = x + dx;

                        if (0 <= ny && ny < height && 0 <= nx && nx < width)
                            f.at<float>(ny, nx) += erro * peso;
                    }
                }
            }
        }
    }
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

    DitheringDifusionErrorDiagonal(img, resultado, dither_map);
    
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

