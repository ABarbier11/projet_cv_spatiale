import yaml
from typing import Dict, Any

def load_config(config_path) -> Dict[str, Any]:
    """
    Charge un fichier de configuration YAML
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        if config is None:
            print(f"Attention: le fichier config {config_path} est vide")
            return {}
        return config
    except FileNotFoundError:
        print(f"Erreur: Fichier config {config_path} non trouvé")
        return {}
    except Exception as e:
        print(f"Erreur lors de la lecture du YAML: {e}")
        return {}