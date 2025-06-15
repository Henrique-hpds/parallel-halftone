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
    std::cout << "Uso: " << program_name << " -i <imagem> -m <modo>\n";
    std::cout << "Modos válidos:\n";
    for (const auto& modo : modos_validos)
        std::cout << "  - " << modo << "\n";
}

void haltone(cv::Mat& img) {
    cv::Mat g = cv::Mat::zeros(img.size(), img.type());

    int height = img.rows;
    int width = img.cols;
    int canales = img.channels();

    
}

int main(int argc, char** argv) {
    std::string img_path;
    std::string modo;
    std::unordered_set<std::string> modos_validos = {"FloydSteinberg", "StevensonArce", "Burkes", "Sierra", "Stucki", "JarvisJudiceNinke"};

    for (int i = 1; i < argc - 1; ++i) {
        std::string flag = argv[i];
        std::string valor = argv[i + 1];

        if (flag == "-i") 
            img_path = valor;
        else if (flag == "-m")
            modo = valor;
    }

    if (img_path.empty() || modo.empty() || modos_validos.find(modo) == modos_validos.end()) {
        print_usage(argv[0], modos_validos);
        return 1;
    }

   std::map<std::pair<int, int>, float> dither_map = create_dither_map(modo);

    std::cout << "Dither map para o modo " << modo << ":\n";
    for (const auto& kv : dither_map) {
        auto [dx, dy] = kv.first;
        std::cout << "(" << dx << ", " << dy << "): " << kv.second << "\n";
    }

    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Erro ao carregar a imagem!" << std::endl;
        return 1;
    }

    haltone(img);

    const std::string win = "Imagem";
    cv::imshow(win, img);

    // Loop: espera por tecla ou fechamento da janela
    while (cv::waitKey(30) < 0 && cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) >= 1);


    return 0;
}

