#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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

typedef struct {
    int dx, dy;
    float peso;
} DitherElement;

typedef struct {
    const char* nome;
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

void dithering_diffusion_diagonal(PPMImage* img, const DitherMetodo* metodo, int grayscale) {
    int width = img->x;
    int height = img->y;
    int size = width * height;

    float* red = (float*)malloc(size * sizeof(float));
    float* green = (float*)malloc(size * sizeof(float));
    float* blue = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        red[i] = img->data[i].red;
        green[i] = img->data[i].green;
        blue[i] = img->data[i].blue;
    }

    for (int diag = 0; diag <= width + height - 2; ++diag) {
        for (int y = 0; y < height; ++y) {
            int x = diag - y;
            if (x < 0 || x >= width) continue;

            int idx = y * width + x;

            float r = red[idx];
            float g = green[idx];
            float b = blue[idx];

            float val = grayscale ? (r * 0.3f + g * 0.59f + b * 0.11f) : 0;

            int r_bin = grayscale ? 0 : (r < 128 ? 0 : 255);
            int g_bin = grayscale ? 0 : (g < 128 ? 0 : 255);
            int b_bin = grayscale ? 0 : (b < 128 ? 0 : 255);
            int gray_bin = grayscale ? (val < 128 ? 0 : 255) : 0;

            float r_err = r - r_bin;
            float g_err = g - g_bin;
            float b_err = b - b_bin;
            float gray_err = val - gray_bin;

            img->data[idx].red = grayscale ? gray_bin : r_bin;
            img->data[idx].green = grayscale ? gray_bin : g_bin;
            img->data[idx].blue = grayscale ? gray_bin : b_bin;

            for (int i = 0; i < metodo->tamanho; i++) {
                int nx = x + metodo->elementos[i].dx;
                int ny = y + metodo->elementos[i].dy;

                if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                    continue;

                int nidx = ny * width + nx;
                float peso = metodo->elementos[i].peso;

                if (grayscale) {
                    float ngray = red[nidx] * 0.3f + green[nidx] * 0.59f + blue[nidx] * 0.11f;
                    ngray += gray_err * peso;
                    red[nidx] = green[nidx] = blue[nidx] = fminf(fmaxf(ngray, 0), 255);
                } else {
                    red[nidx] = fminf(fmaxf(red[nidx] + r_err * peso, 0), 255);
                    green[nidx] = fminf(fmaxf(green[nidx] + g_err * peso, 0), 255);
                    blue[nidx] = fminf(fmaxf(blue[nidx] + b_err * peso, 0), 255);
                }
            }
        }
    }

    free(red);
    free(green);
    free(blue);
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
    writePPM(img, "output.ppm");

    free(img->data);
    free(img);
    return 0;
}
