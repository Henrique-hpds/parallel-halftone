# Projeto Final MC970 - Paralelização de Técnicas de Meios-Tons

Este projeto implementa algoritmos de meios-tons com difusão de erro em três versões: serial (C++), paralela com OpenMP (C++) e paralela com CUDA (C++/CUDA). Também inclui scripts Python para benchmarking e análise dos resultados.

---

## 1. Pré-requisitos

- **Linux** (recomendado)
- **Compilador C++** (g++ >= 9)
- **CUDA Toolkit** (para rodar a versão CUDA)
- **Python 3.8+**
- **CMake** (>= 3.10)
- **make**
- **pip** (>= 20)
- **NVIDIA GPU** (para CUDA)
- **Google Colab** (opcional, para rodar o notebook)

---

## 2. Baixando o Projeto

Clone este repositório via ssh:

```bash
git clone git@github.com:mc970-25s1/final-project-paralelizacao-tecnica-de-meios-tons.git
cd final-project-paralelizacao-tecnica-de-meios-tons
```

---

## 3. Instalando Dependências C++/CUDA

No Ubuntu, rode:

```bash
sudo apt update
sudo apt install build-essential cmake libomp-dev nvidia-cuda-toolkit
```

Certifique-se de seguir as orientações da nvidia para configurar o CUDA.

---

## 4. Compilando o Projeto

Crie o diretório de build e compile:

```bash
cmake -S . -B build
cmake --build build -j
```

Os executáveis serão gerados em `build/`.

---

## 5. Executando os Programas

### Serial

```bash
./build/serial_cpp <input.ppm> <output.ppm> <method> [p=0.5] [stochastic=1] [-g]
```

### OpenMP

```bash
./build/omp <input.ppm> <output.ppm> <method> [p=0.5] [stochastic=1] [-g]
```

### CUDA

```bash
./build/cuda <input.ppm> <output.ppm> <method> [p=0.5] [stochastic=1] [-g]
```

**Parâmetros:**
- `<input.ppm>`: Caminho para a imagem de entrada (formato PPM).
- `<output.ppm>`: Caminho para a imagem de saída.
- `<method>`: Método de difusão de erro (`FloydSteinberg`, `StevensonArce`, `Burkes`, `Sierra`, `Stucki`, `JarvisJudiceNinke`).
- `[p]`: Parâmetro de ruído para dithering estocástico (opcional, padrão 0.5).
- `[stochastic]`: 1 para estocástico, 0 para determinístico (opcional, padrão 1).
- `[-g]`: Converte para escala de cinza antes de processar (opcional).

**Exemplo:**

```bash
./build/omp img/ppm/teste.ppm out/teste_out.ppm FloydSteinberg 0.5 1 -g
```

---

## 6. Ambiente Python para Benchmark e Análise

### Criando o Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Instalando as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Rodando o Notebook Localmente ou no Google Colab

1. Faça upload do projeto para o seu Google Drive.
2. Abra o arquivo `colab-runner.ipynb` no Google Colab.
3. Execute as células na ordem. O notebook monta o Google Drive, compila o projeto, executa os testes e gera relatórios de desempenho automaticamente.

---

## 8. Estrutura de Pastas

- `src/` — Códigos-fonte C++/CUDA
- `img/ppm/` — Imagens de entrada (PPM)
- `out/` — Imagens de saída
- `build/` — Executáveis compilados
- `logs/` — Logs e relatórios de execução
- `colab-runner.ipynb` — Notebook para automação e análise

---

## 9. Observações

- Para rodar a versão CUDA, é necessário ter uma GPU NVIDIA compatível e drivers/CUDA instalados.
- O notebook pode ser executado localmente ou no Colab, mas a versão CUDA só roda em ambientes com GPU NVIDIA.
- As imagens de entrada devem estar no formato PPM.

## 10. Autores

- **Henrique Parede de Souza** [[Github](https://github.com/Henrique-hpds)] [[Linkedin](https://www.linkedin.com/in/henrique-parede-de-souza/)]
- **Raphael Salles Vitor de Souza** [[Github](https://github.com/RaphaelSVSouza)] [[Linkedin](https://www.linkedin.com/in/raphael-sv-souza/)]
- **Vinicius Patriarca Miranda Miguel** [[Github](https://github.com/viniciuskant)] [[Linkedin](https://www.linkedin.com/in/vinicius-patriarca-miranda-miguel-01b568236/)]

## 11. Como citar

```bibtex
@software{Souza2025,
  title    = "{complexity-of-UNO}",
  author   = "Souza, Henrique Parede de and Souza, Raphael Salles Vitor de and Miguel, Vinicius Patriarca Miranda",
  month    =  jun,
  year     =  2025,
  language = "pt"
}
```
