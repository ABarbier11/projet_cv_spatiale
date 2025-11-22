import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List

def get_pairs_dataframe(
    config: Dict[str, Any], 
    project_root: Path, 
    categories: Union[str, List[str]] = "ALL"
) -> pd.DataFrame:
    """
    Charge les paires d'images pour les catégories spécifiées.
    
    Args:
        config: Le dictionnaire de configuration.
        project_root: La racine du projet (Path).
        categories: "ALL" pour tout charger, ou une liste ["agri", "urban"], ou un str "agri".
    """
    # 1. Récupération de la racine des données
    data_root_rel = config['dataset']['root_dir']
    data_root = project_root / data_root_rel
    
    # 2. Détermination des catégories à charger
    available_cats = config['dataset']['available_categories']
    
    target_cats = []
    if categories == "ALL":
        target_cats = available_cats
    elif isinstance(categories, str):
        target_cats = [categories]
    elif isinstance(categories, list):
        target_cats = categories
    else:
        raise ValueError("L'argument 'categories' doit être 'ALL', un str ou une liste de str.")

    # 3. Boucle sur chaque catégorie pour collecter les données
    all_dataframes = []
    
    s1_name = config['dataset']['s1_folder']
    s2_name = config['dataset']['s2_folder']
    
    sar_col = config['columns']['sar']
    opt_col = config['columns']['opt']
    cat_col = config['columns']['category']

    print(f"Chargement des données pour : {target_cats}")

    for cat in target_cats:
        # Construction des chemins : project_root/data/agri/s1
        cat_dir = data_root / cat
        s1_dir = cat_dir / s1_name
        s2_dir = cat_dir / s2_name
        
        if not s1_dir.exists() or not s2_dir.exists():
            print(f"⚠️  Dossier introuvable pour '{cat}' (S1 ou S2 manquant). Ignoré.")
            continue
        
        s1_files = sorted([str(p) for p in s1_dir.glob('*.png')])
        s2_files = sorted([str(p) for p in s2_dir.glob('*.png')])

        if len(s1_files) != len(s2_files):
            min_len = min(len(s1_files), len(s2_files))
            print(f"⚠️  Mismatch dans '{cat}' : S1={len(s1_files)}, S2={len(s2_files)}. Tronqué à {min_len}.")
            s1_files = s1_files[:min_len]
            s2_files = s2_files[:min_len]

        # Création du petit DataFrame pour cette catégorie
        df_cat = pd.DataFrame({
            sar_col: s1_files,
            opt_col: s2_files,
            cat_col: cat
        })
        
        all_dataframes.append(df_cat)

    final_df = pd.concat(all_dataframes, ignore_index=True)
    return final_df