#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define IDX(x, y, width) ((y) * (width) + (x))
#define RGB_COMPONENT_COLOR 255
#define MAX_DITHER_SIZE 20

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;


static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

void writePPM(PPMImage *img, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Não foi possível abrir o arquivo de saída '%s'\n", filename);
        exit(1);
    }
    fprintf(fp, "P6\n");
    fprintf(fp, "# %s\n", "Dithering");
    fprintf(fp, "%d %d\n", img->x, img->y);
    fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

struct DitherElement {
    int dx, dy;
    float peso;
};

typedef struct {
    const char* nome;   // não posso usar nome dentro do kernel CUDA
    DitherElement elementos[MAX_DITHER_SIZE];
    int tamanho;
} DitherMetodo;


DitherMetodo getDitherMetodo(const char* nome) {
    DitherMetodo metodo = { nome, {{0}}, 0 };

    if (strcmp(nome, "FloydSteinberg") == 0) {
        metodo = (DitherMetodo){ nome,
            {
                {1, 0, 7.f/16}, {-1, 1, 3.f/16}, {0, 1, 5.f/16}, {1, 1, 1.f/16}
            }, 4
        };
    } else if (strcmp(nome, "Burkes") == 0) {
        metodo = (DitherMetodo){ nome,
            {
                {1, 0, 8.f/32}, {2, 0, 4.f/32},
                {-2, 1, 2.f/32}, {-1, 1, 4.f/32}, {0, 1, 8.f/32},
                {1, 1, 4.f/32}, {2, 1, 2.f/32}
            }, 7
        };
    }
    // Adicione os outros métodos aqui...

    return metodo;
}

// Kernel CUDA para converter imagem para escala de cinza
__global__ void convertToGrayscaleKernel(PPMPixel* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = IDX(x, y, width);
        unsigned char gray = (unsigned char)(0.299f * pixels[idx].red + 
                                            0.587f * pixels[idx].green + 
                                            0.114f * pixels[idx].blue);
        pixels[idx].red = gray;
        pixels[idx].green = gray;
        pixels[idx].blue = gray;
    }
}

// Kernel CUDA para aplicar dithering diagonal
__global__ void ditheringDiagonalKernel(PPMPixel* pixels, int width, int height, 
                                       DitherElement* ditherElements, int numElements) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = IDX(x, y, width);

        // Cor original
        float oldColor[3] = {
            (float)pixels[idx].red,
            (float)pixels[idx].green,
            (float)pixels[idx].blue
        };

        // Cor quantizada e erro de quantização
        float newColor[3];
        float quantizationError[3];

        // 1. Calcule a cor quantizada (0 ou 255)
        for (int c = 0; c < 3; c++) {
            newColor[c] = (oldColor[c] > 128) ? 255.0f : 0.0f; // Quantização em 2 tons (preto ou branco)
            quantizationError[c] = oldColor[c] - newColor[c]; // Erro de quantização
        }

        // 2. Propague o erro para os vizinhos
        for (int i = 0; i < numElements; i++) {
            DitherElement element = ditherElements[i];
            int nx = x + element.dx;
            int ny = y + element.dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = IDX(nx, ny, width);
                for (int c = 0; c < 3; c++) {
                    unsigned char* channel = (c == 0) ? &pixels[nidx].red :
                                            (c == 1) ? &pixels[nidx].green :
                                                       &pixels[nidx].blue;

                    // Ajustar o canal de cor do pixel vizinho com o erro ponderado
                    float newValue = (float)(*channel) + quantizationError[c] * element.peso;
                    newValue = fminf(255.0f, fmaxf(0.0f, newValue)); // Limitar entre 0 e 255
                    *channel = (unsigned char)newValue; // Atualizar o valor do pixel vizinho
                }
            }
        }

        // 3. Definir o valor final para o pixel atual após o dithering
        pixels[idx].red   = (unsigned char)fminf(255.0f, fmaxf(0.0f, newColor[0]));
        pixels[idx].green = (unsigned char)fminf(255.0f, fmaxf(0.0f, newColor[1]));
        pixels[idx].blue  = (unsigned char)fminf(255.0f, fmaxf(0.0f, newColor[2]));
    }
}


void run_dithering_diagonal_cuda(PPMImage* img, DitherMetodo* metodo, int grayscale) {
  PPMPixel* d_pixels = nullptr;
  DitherElement* d_ditherElements = nullptr;
  
  // Alocar memória na GPU
  size_t imageSize = img->x * img->y * sizeof(PPMPixel);
  size_t ditherSize = metodo->tamanho * sizeof(DitherElement);
  
  cudaError_t err;
  err = cudaMalloc((void**)&d_pixels, imageSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Erro ao alocar memória para pixels na GPU: %s\n", cudaGetErrorString(err));
    return;
  }
  
  err = cudaMalloc((void**)&d_ditherElements, ditherSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Erro ao alocar memória para elementos de dithering na GPU: %s\n", cudaGetErrorString(err));
    cudaFree(d_pixels);
    return;
  }
  
  // Copiar dados para a GPU
  err = cudaMemcpy(d_pixels, img->data, imageSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Erro ao copiar pixels para a GPU: %s\n", cudaGetErrorString(err));
    cudaFree(d_pixels);
    cudaFree(d_ditherElements);
    return;
  }
  
  err = cudaMemcpy(d_ditherElements, metodo->elementos, ditherSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Erro ao copiar elementos de dithering para a GPU: %s\n", cudaGetErrorString(err));
    cudaFree(d_pixels);
    cudaFree(d_ditherElements);
    return;
  }
  
  // Configurar dimensões do grid e blocos
  dim3 blockSize(16, 16);
  dim3 gridSize((img->x + blockSize.x - 1) / blockSize.x, 
          (img->y + blockSize.y - 1) / blockSize.y);
  
  // Converter para escala de cinza se necessário
  if (grayscale) {
    convertToGrayscaleKernel<<<gridSize, blockSize>>>(d_pixels, img->x, img->y);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "Erro ao executar kernel de conversão para escala de cinza: %s\n", cudaGetErrorString(err));
      cudaFree(d_pixels);
      cudaFree(d_ditherElements);
      return;
    }
  }
  
  // Aplicar dithering diagonal
  ditheringDiagonalKernel<<<gridSize, blockSize>>>(d_pixels, img->x, img->y, 
                          d_ditherElements, metodo->tamanho);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Erro ao executar kernel de dithering diagonal: %s\n", cudaGetErrorString(err));
    cudaFree(d_pixels);
    cudaFree(d_ditherElements);
    return;
  }
  
  // Copiar resultado de volta para a CPU
  err = cudaMemcpy(img->data, d_pixels, imageSize, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Erro ao copiar pixels da GPU para a CPU: %s\n", cudaGetErrorString(err));
  }
  
  // Liberar memória da GPU
  cudaFree(d_pixels);
  cudaFree(d_ditherElements);
}

// Função para substituir a função original (mantida para compatibilidade)
void dithering_diffusion_diagonal(PPMImage* img, DitherMetodo* metodo, int grayscale) {
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    run_dithering_diagonal_cuda(img, metodo, grayscale);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tempo de execução da função (GPU): %.3f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printf("Uso: %s -i <imagem.ppm> -m <modo> [-g]\n", argv[0]);
        return 1;
    }

    const char* img_path = NULL;
    const char* metodo_nome = NULL;
    int grayscale = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            img_path = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            metodo_nome = argv[++i];
        } else if (strcmp(argv[i], "-g") == 0) {
            grayscale = 1;
        }
    }

    if (!img_path || !metodo_nome) {
        printf("Parâmetros inválidos.\n");
        return 1;
    }

    PPMImage* img = readPPM(img_path);
    DitherMetodo metodo = getDitherMetodo(metodo_nome);

    if (metodo.tamanho == 0) {
        fprintf(stderr, "Método '%s' não reconhecido.\n", metodo_nome);
        return 1;
    }

    dithering_diffusion_diagonal(img, &metodo, grayscale);

    char output_path[256];
    strncpy(output_path, img_path, sizeof(output_path) - 1);
    output_path[sizeof(output_path) - 1] = '\0';

    // Alterar extensão para "_cu.ppm"
    char* dot = strrchr(output_path, '.');
    if (dot) {
        strcpy(dot, "_cu.ppm");
    } else {
        strcat(output_path, "_cu.ppm");
    }

    // Trocar o diretório "img/" por "out/"
    char* img_dir = strstr(output_path, "img/");
    if (img_dir) {
        memmove(img_dir + 4, img_dir + 4, strlen(img_dir + 4) + 1);  // desloca o restante para manter o tamanho
        memcpy(img_dir, "out/", 4);  // sobrescreve "img/" por "out/"
    }

    printf("Salvando em: %s\n", output_path);
    writePPM(img, output_path);


    free(img->data);
    free(img);
    return 0;
}
