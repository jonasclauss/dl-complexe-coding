import argparse
import json
import os
from helpers.logger import setup_logger

logger = setup_logger()

# Default-Konfiguration im Code (Fallback, wenn keine config-Datei und keine CLI-Argumente)
DEFAULT_CONFIG = {
    "seed": 3792567,
    "project_path": ".",
    # Vollständiger Pfad zum Datensatz-Ordner (z.B. "./data/EuroSAT_RGB" oder "./data/EuroSAT_MS")
    "data_path": "./data/EuroSAT_RGB",
    "data_source": "rgb",  # "rgb" | "ms" (steuert nur, welches Netz/Preprocessing verwendet wird)
    "epochs": 15,
    "batch_size": 128,
    "workers": 6,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    # Liste von Augmentations-/Preprocessing-Tags, z.B. ["mild", "resnet"]
    # Unterstützt: "none", "mild", "strong", "resnet"
    # "none" = keine Transformation (überschreibt andere),
    # "resnet" = ResNet-Normalisierung (nur RGB+ResNet sinnvoll)
    "augmentation": ["resnet"],
    # Modellwahl: eigenes CNN oder pretrained ResNet18 (für RGB/MS unterschiedlich gemappt)
    "model": "resnet",  # "cnn" | "resnet"
    "reproduction": False,
    "save_logits": False,
    "logits_path": "logits.pt",
    "model_path": "model.pth",
}


def load_config(args: argparse.Namespace) -> dict:
    """Lade Konfiguration aus Defaults, optionaler JSON-Config und CLI-Overrides."""
    config = DEFAULT_CONFIG.copy()

    # 1) Versuche, eine Config-Datei zu laden (explizit über --config oder implizit config.json)
    config_path = getattr(args, "config", None)
    if config_path is None:
        default_path = "config.json"
        if os.path.exists(default_path):
            config_path = default_path

    if config_path is not None and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            if isinstance(file_cfg, dict):
                for k, v in file_cfg.items():
                    if v is not None and k in config:
                        config[k] = v
        except Exception as e:
            logger.warning(f"Could not load config file '{config_path}': {e}")

    # 2) CLI-Overrides (haben Vorrang vor Datei und Defaults)
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed
    if getattr(args, "data_path", None) is not None:
        config["data_path"] = args.data_path
    if getattr(args, "project_path", None) is not None:
        config["project_path"] = args.project_path
    if getattr(args, "epochs", None) is not None:
        config["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        config["batch_size"] = args.batch_size
    if getattr(args, "workers", None) is not None:
        config["workers"] = args.workers
    if getattr(args, "lr", None) is not None:
        config["learning_rate"] = args.lr
    if getattr(args, "weight_decay", None) is not None:
        config["weight_decay"] = args.weight_decay
    if getattr(args, "augmentation", None) is not None:
        config["augmentation"] = args.augmentation
    if getattr(args, "data_source", None) is not None:
        config["data_source"] = args.data_source
    if getattr(args, "reproduction", None) is not None:
        config["reproduction"] = args.reproduction
    if getattr(args, "save_logits", None) is not None:
        config["save_logits"] = args.save_logits
    if getattr(args, "logits_path", None) is not None:
        config["logits_path"] = args.logits_path
    if getattr(args, "model_path", None) is not None:
        config["model_path"] = args.model_path
    if getattr(args, "model", None) is not None:
        config["model"] = args.model

    return config
