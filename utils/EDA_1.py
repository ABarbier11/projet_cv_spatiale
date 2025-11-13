import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def show_image_pair(sar_path: str, opt_path: str) -> None:
    """
    Charge et affiche une paire d'image SAR et OPTIQUE
    """
    sar_img = Image.open(sar_path)
    opt_img = Image.open(opt_path)

    print(f"Size SAR : {sar_img.size}, Canaux : {sar_img.mode}")
    print(f"Size OPT : {opt_img.size}, Canaux: {opt_img.mode}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    ax1.imshow(sar_img, cmap='gray')
    ax1.set_title("Image SAR")

    ax2.imshow(opt_img)
    ax2.set_title("Image OPT")

    plt.suptitle('Alignement visuel des images')
    plt.tight_layout()
    plt.show()

def sample_random_pairs(df_pairs: pd.DataFrame, config_dataset: Dict[str, Any]) -> Tuple[str, str]:
    """
    Prend le DataFrame des pairs (et la config des noms de colonnes) et renvoie le tuple d'une paire au hasard
    """

    sar_col = config_dataset['data_columns']['sar']
    opt_col = config_dataset['data_columns']['opt']

    sample = df_pairs.sample(1).iloc[0]
    return sample[sar_col], sample[opt_col]