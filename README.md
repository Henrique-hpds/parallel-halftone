[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/fH_qNtDu)


# Projeto: Paralelização da Técnica de Meios-Tons (Floyd-Steinberg)

Este projeto implementa a técnica de dithering de Floyd-Steinberg para imagens PPM, com versões serial, paralela (OpenMP) e CUDA.

## Códigos e Descrição

### serial.cpp
Implementação serial do algoritmo de Floyd-Steinberg, com opção estocástica.

### omp.cpp
Implementação paralela usando OpenMP, também com opção estocástica.

### cuda.cu (executável: parallel)
Implementação CUDA do algoritmo, com suporte a modo padrão e estocástico.

---

## Como rodar e exemplos de uso

Todos os programas esperam imagens no formato PPM (P6).

### Parâmetros e flags
- `<input.ppm>`: Imagem de entrada
- `<output.ppm>`: Caminho para salvar a imagem de saída
- `[p=0.5]`: (Opcional) Parâmetro de peso para o ruído estocástico (default: 0.5)
- `[stochastic=1]`: (Opcional) 1 para modo estocástico, 0 para Floyd-Steinberg clássico (default: 1)
- `-g`: (Opcional) Converte a imagem para escala de cinza antes de processar

### Exemplos Utilizados para execução

#### Serial
```
./serial ../img/vazio_roxo.ppm ../out/serial_normal.ppm 0.5 0
./serial ../img/vazio_roxo.ppm ../out/serial_stochastic.ppm 0.5 1
```

#### OpenMP
```
./omp ../img/vazio_roxo.ppm ../out/omp_normal.ppm 0.5 0
./omp ../img/vazio_roxo.ppm ../out/omp_stochastic.ppm 0.5 1
```

#### CUDA
```
./cuda ../img/vazio_roxo.ppm ../out/cuda_normal.ppm 5 0
./cuda ../img/vazio_roxo.ppm ../out/cuda_stochastic.ppm 5 1
```

#### Opção para escala de cinza
```
./serial ../img/vazio_roxo.ppm ../out/serial_gray.ppm 0.5 1 -g
./omp ../img/vazio_roxo.ppm ../out/omp_gray.ppm 0.5 1 -g
./cuda ../img/vazio_roxo.ppm ../out/cuda_gray.ppm 0.5 1 -g
```

---

## Observações
- O parâmetro `p` controla a intensidade do ruído estocástico na difusão de erro.
- O parâmetro `stochastic` ativa (`1`) ou desativa (`0`) o modo estocástico.
- As imagens de saída são salvas no caminho especificado pelo usuário.
- Para compilar, utilize o CMake fornecido no projeto.
