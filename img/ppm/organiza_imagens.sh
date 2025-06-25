#!/bin/bash

# Cria pastas, se necessário
mkdir -p small medium large xlarge

# Mover imagens com base no tamanho, se existirem arquivos .ppm
ppm_files=(*.ppm)
if [ -e "${ppm_files[0]}" ]; then
    for img in *.ppm; do
        size=$(file "$img" | grep -oP '\d+ x \d+')
        width=$(echo "$size" | cut -d 'x' -f 1 | tr -d ' ')
        height=$(echo "$size" | cut -d 'x' -f 2 | tr -d ' ')

        if [[ "$width" -le 256 && "$height" -le 256 ]]; then
            mv "$img" small/
        elif [[ "$width" -le 512 && "$height" -le 512 ]]; then
            mv "$img" medium/
        elif [[ "$width" -le 1024 && "$height" -le 1024 ]]; then
            mv "$img" large/
        else
            mv "$img" xlarge/
        fi
    done
fi

# Renomear imagens nas pastas small, medium, large e xlarge
for categoria in small medium large xlarge; do
    pasta="$categoria"
    if [ -d "$pasta" ]; then
        contador=1
        for imagem in "$pasta"/*.ppm; do
            # Se não houver arquivos .ppm, continue
            [ -e "$imagem" ] || continue
            extensao="${imagem##*.}"
            novo_nome="${pasta}_${contador}.${extensao}"
            mv "$imagem" "$pasta/$novo_nome"
            contador=$((contador + 1))
        done
    fi
done

