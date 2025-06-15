#include <opencv2/opencv.hpp>
#include <unordered_set>
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
void haltone(cv::Mat& img, std::map<std::pair<int, int>, float>& kernel) {
    int height = img.rows;
    int width = img.cols;
    int channels = img.channels();

    cv::Mat saida;
    cv::Mat imagem_float;

    if (channels == 1) {
        img.convertTo(imagem_float, CV_32FC1);
        saida = cv::Mat::zeros(height, width, CV_8UC1);

        for (int y = 0; y < height; ++y) {
            bool esquerda_para_direita = (y % 2 == 0);
            int x_start = esquerda_para_direita ? 0 : width - 1;
            int x_end   = esquerda_para_direita ? width : -1;
            int passo   = esquerda_para_direita ? 1 : -1;

            for (int x = x_start; x != x_end; x += passo) {
                float valor_pixel = imagem_float.at<float>(y, x);
                uint8_t novo_valor = (valor_pixel < 128.0f) ? 0 : 255;
                saida.at<uchar>(y, x) = novo_valor;

                float erro = valor_pixel - novo_valor;

                for (const auto& [offset, peso] : kernel) {
                    int dx = offset.second;
                    int dy = offset.first;
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        imagem_float.at<float>(ny, nx) += erro * peso;
                    }
                }
            }
        }
    } else {
        img.convertTo(imagem_float, CV_32FC3);
        saida = cv::Mat::zeros(height, width, CV_8UC3);

        std::vector<cv::Mat> canais(3);
        cv::split(imagem_float, canais);
        std::vector<cv::Mat> canais_saida(3, cv::Mat::zeros(height, width, CV_8UC1));

        for (int c = 0; c < 3; ++c) {
            cv::Mat& f = canais[c];
            cv::Mat& g = canais_saida[c];

            for (int y = 0; y < height; ++y) {
                bool esquerda_para_direita = (y % 2 == 0);
                int x_start = esquerda_para_direita ? 0 : width - 1;
                int x_end   = esquerda_para_direita ? width : -1;
                int passo   = esquerda_para_direita ? 1 : -1;

                for (int x = x_start; x != x_end; x += passo) {
                    float valor_pixel = f.at<float>(y, x);
                    uint8_t novo_valor = (valor_pixel < 128.0f) ? 0 : 255;
                    g.at<uchar>(y, x) = novo_valor;

                    float erro = valor_pixel - novo_valor;

                    for (const auto& [offset, peso] : kernel) {
                        int dx = offset.second;
                        int dy = offset.first;
                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            f.at<float>(ny, nx) += erro * peso;
                        }
                    }
                }
            }
        }
        cv::merge(canais_saida, saida);
    }

    cv::imshow("Imagem", saida);
    cv::waitKey(0);
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

    std::map<std::pair<int, int>, float> dither_map = create_dither_map(modo);
    
    cv::Mat img;
    if (grayscale)
        img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    else
        img = cv::imread(img_path, cv::IMREAD_COLOR);
    
    // img.convertTo(img, CV_32F);

    if (img.empty()) {
        std::cerr << "Erro ao carregar a imagem!" << std::endl;
        return 1;
    }

    haltone(img, dither_map);

    const std::string win = "Imagem";
    cv::imshow(win, img);

    // Loop: espera por tecla ou fechamento da janela
    while (cv::waitKey(30) < 0 && cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) >= 1);

    return 0;
}

