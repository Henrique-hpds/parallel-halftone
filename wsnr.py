import sys
import numpy as np
from PIL import Image

def wsnr(img1, img2):
    """Calcula o WSNR entre duas imagens numpy."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    signal_power = np.mean(img1 ** 2)
    return 10 * np.log10(signal_power / mse)

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "-g":
        mode = 'L'
        img1_path = sys.argv[2]
        img2_path = sys.argv[3]
    elif len(sys.argv) == 3:
        mode = 'RGB'
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
    else:
        print("Uso: python wsnr.py [-g] <imagem1> <imagem2>")
        sys.exit(1)

    img1 = Image.open(img1_path).convert(mode)
    img2 = Image.open(img2_path).convert(mode)
    score = wsnr(img1, img2)
    print(f"WSNR: {score:.4f} dB")