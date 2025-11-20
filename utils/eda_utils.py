import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from tqdm import tqdm


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

def plot_dataset_histograms(df_pairs: pd.DataFrame, config_dataset: Dict[str, Any], sample_size: int = 100) -> None:
    """
    Analyse un échantillon du dataset et affiche les histogrammes de 
    distribution des pixels pour les images SAR et Optiques.
    Calcule également les statistiques de moyenne et d'écart-type.
    """
    
    print(f"Analyse de la distribution sur {sample_size} paires aléatoires...")
    
    sar_col = config_dataset['columns']['sar']
    opt_col = config_dataset['columns']['optical']

    sar_pixels = []
    opt_r_pixels = []
    opt_g_pixels = []
    opt_b_pixels = []

    df_sample = df_pairs.sample(n=min(sample_size, len(df_pairs)), replace=False)

    for _, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0], desc="Chargement des images"):
        sar_img = Image.open(row[sar_col])
        opt_img = Image.open(row[opt_col])
        
        sar_np = np.array(sar_img)
        opt_np = np.array(opt_img)
        
        # .ravel() "aplatit" l'image en un long vecteur 1D
        sar_pixels.append(sar_np.ravel())
        
        # Séparer les canaux optiques
        opt_r_pixels.append(opt_np[..., 0].ravel())
        opt_g_pixels.append(opt_np[..., 1].ravel())
        opt_b_pixels.append(opt_np[..., 2].ravel())

    sar_all = np.concatenate(sar_pixels)
    opt_r_all = np.concatenate(opt_r_pixels)
    opt_g_all = np.concatenate(opt_g_pixels)
    opt_b_all = np.concatenate(opt_b_pixels)

    print("\n--- Statistiques (Moyenne, Écart-type) ---")
    print(f"[SAR]   Moy: {np.mean(sar_all):.3f}, Std: {np.std(sar_all):.3f}, Min: {np.min(sar_all):.3f}, Max: {np.max(sar_all):.3f}")
    print(f"[OPT R] Moy: {np.mean(opt_r_all):.3f}, Std: {np.std(opt_r_all):.3f}, Min: {np.min(opt_r_all):.3f}, Max: {np.max(opt_r_all):.3f}")
    print(f"[OPT G] Moy: {np.mean(opt_g_all):.3f}, Std: {np.std(opt_g_all):.3f}, Min: {np.min(opt_g_all):.3f}, Max: {np.max(opt_g_all):.3f}")
    print(f"[OPT B] Moy: {np.mean(opt_b_all):.3f}, Std: {np.std(opt_b_all):.3f}, Min: {np.min(opt_b_all):.3f}, Max: {np.max(opt_b_all):.3f}")

    # --- Tracé des Histogrammes ---
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Histogramme SAR ---
    # Nous utilisons une échelle log sur l'axe Y pour mieux voir la "longue traîne"
    ax1.hist(sar_all, bins=100, color='blue', alpha=0.7, log=True)
    ax1.set_title("Distribution des Pixels SAR (S1) - (Échelle Y Log)")
    ax1.set_xlabel("Intensité Pixel")
    ax1.set_ylabel("Fréquence (Log)")

    # --- Histogramme Optique (RGB) ---
    ax2.hist(opt_r_all, bins=100, color='red', alpha=0.5, label='Canal Rouge', density=True)
    ax2.hist(opt_g_all, bins=100, color='green', alpha=0.5, label='Canal Vert', density=True)
    ax2.hist(opt_b_all, bins=100, color='blue', alpha=0.5, label='Canal Bleu', density=True)
    ax2.set_title("Distribution des Pixels Optiques (S2) - (Canaux RGB)")
    ax2.set_xlabel("Intensité Pixel (0-255 ?)")
    ax2.set_ylabel("Densité")
    ax2.legend()
    
    plt.suptitle("Analyse de la Distribution des Pixels (sur l'échantillon)", fontsize=16)
    plt.tight_layout()
    plt.show()