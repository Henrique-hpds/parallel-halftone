[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/fH_qNtDu)

# Compilar e Executar Projeto

Para compilar o projeto, inicie um **terminal que não seja o do vscode** e rode a partir da raiz do repositório:

```bash
mkdir build
cd build
cmake ..
make
```

Para executar o projeto após a compilação:

```bash
./serial -i <path_to_image> -m <method> [-g]
```

Exemplos de execução

```bash
./serial -i ../img/vazio_roxo.png -m Stucki
```

- -i: <imagem>    Caminho para a imagem de entrada
- -m: <modo>      Modo de dithering a ser utilizado
- -g:             (Opcional) Ativa o modo escala de cinza

Modos válidos:
  - JarvisJudiceNinke
  - Stucki
  - Sierra
  - Burkes
  - StevensonArce
  - FloydSteinberg
