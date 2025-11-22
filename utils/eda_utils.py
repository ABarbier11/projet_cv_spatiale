import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from tqdm import tqdm
from pathlib import Path


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

    sar_col = config_dataset['columns']['sar']
    opt_col = config_dataset['columns']['opt']

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
    opt_col = config_dataset['columns']['opt']

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


def plot_image_gallery(
    df_pairs: pd.DataFrame, 
    config: Dict[str, Any], 
    n_rows: int = 3, 
    n_cols: int = 3,
    seed: int = None
) -> None:
    """
    Affiche une galerie d'images (SAR à gauche, Optique à droite) pour vérifier 
    visuellement la diversité, les nuages, et la cohérence des saisons.
    
    Args:
        df_pairs: Le DataFrame contenant les chemins.
        config: La configuration pour les noms de colonnes.
        n_rows: Nombre de lignes de la grille.
        n_cols: Nombre de PAIRES par ligne.
        seed: Pour reproduire le même échantillon (optionnel).
    """
    n_pairs = n_rows * n_cols

    sample = df_pairs.sample(n=n_pairs, random_state=seed)
    
    sar_col = config['columns']['sar']
    opt_col = config['columns']['opt']

    # On multiplie n_cols par 2 car chaque "item" est une paire (SAR + Opt)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 5, n_rows * 2.5))
    
    fig.suptitle(f"Galerie Diversité : SAR (Gauche) vs Optique (Droite) - n={n_pairs}", fontsize=16, y=0.98)
    axes = axes.flatten()
    
    for idx, (i, row) in enumerate(sample.iterrows()):
        # idx * 2 = l'emplacement SAR
        # idx * 2 + 1 = l'emplacement Optique
        ax_sar = axes[idx * 2]
        ax_opt = axes[idx * 2 + 1]
        
        sar_path = row[sar_col]
        opt_path = row[opt_col]
        
        sar_img = Image.open(sar_path)
        opt_img = Image.open(opt_path)
        
        # Affichage SAR
        ax_sar.imshow(sar_img, cmap='gray')
        ax_sar.axis('off')
        short_name = Path(sar_path).name[:15] + "..."
        ax_sar.set_title(f"SAR\n{short_name}", fontsize=8, color='#333333')
        
        # Affichage OPT
        ax_opt.imshow(opt_img)
        ax_opt.axis('off')
        ax_opt.set_title("OPT", fontsize=8, color='#333333')
            

    plt.tight_layout()
    plt.show()