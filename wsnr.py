import os
import numpy as np
from PIL import Image
from collections import defaultdict

def wsnr(img1, img2):
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    signal_power = np.mean(img1 ** 2)
    return 10 * np.log10(signal_power / mse)

def strip_suffix(filename, method):
    suffix = f"_{method}_no_stochastic"
    return filename.replace(suffix, f"_{method}") if suffix in filename else filename

def process_images(base_dir, output_file):
    with open(output_file, 'w') as f:
        methods = ["cuda", "openmp", "serial"]

        for method in methods:
            no_stochastic_dir = os.path.join(base_dir, f"{method}_no_stochastic")
            stochastic_dir = os.path.join(base_dir, f"{method}_stochastic")

            if not (os.path.exists(no_stochastic_dir) and os.path.exists(stochastic_dir)):
                print(f"[!] Diretórios para {method} não encontrados.")
                continue

            print(f"\n======= Método: {method} =======", file=f)
            print(f"\n======= Método: {method} =======")

            # Dicionário para agrupar por tamanho (small, medium, etc.)
            grouped_scores = defaultdict(list)
            all_scores = []

            for root, _, files in os.walk(no_stochastic_dir):
                for file in sorted(files):
                    if not file.endswith(".ppm"):
                        continue

                    img1_path = os.path.join(root, file)
                    relative_subdir = os.path.relpath(root, no_stochastic_dir)
                    img2_dir = os.path.join(stochastic_dir, relative_subdir)
                    img2_filename = strip_suffix(file, method)
                    img2_path = os.path.join(img2_dir, img2_filename)

                    if os.path.exists(img2_path):
                        img1 = Image.open(img1_path).convert('RGB')
                        img2 = Image.open(img2_path).convert('RGB')
                        score = wsnr(img1, img2)
                        all_scores.append(score)

                        folder_name = os.path.basename(relative_subdir)
                        grouped_scores[folder_name].append((file, score))
                    else:
                        print(f"[!] Correspondente não encontrado: {img2_path}")

            # Imprimir resultados agrupados por tamanho
            for size in sorted(grouped_scores):
                print(f"--- {size} ---", file=f)
                print(f"--- {size} ---")
                for filename, score in grouped_scores[size]:
                    filename = filename.replace("_serial_cpp", "").replace("_openmp_no_stochastic", "").replace("cuda_no_stochastic", "")
                    line = f"{filename:<20} WSNR = {score:.4f} dB"
                    print(line)
                    f.write(line + "\n")
                print("", file=f)

            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                summary = f"Média WSNR ({method}): {avg_score:.4f} dB"
                print(summary + "\n")
                f.write(summary + "\n\n")
            else:
                print(f"[!] Nenhuma imagem comparada para método: {method}")

if __name__ == "__main__":
    base_dir = "out"
    output_file = "wsnr_results.txt"
    process_images(base_dir, output_file)
