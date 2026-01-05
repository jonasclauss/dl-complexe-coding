import argparse
import os
import random
from helpers.dataset import DataReader
from helpers.nn import NeuralNetwork, NeuralNetworkMS, PretrainedResNet18, PretrainedResNet18MS
from helpers.config import load_config
from helpers.training import set_fixed_seed, train, evaluate
from helpers.plotting import analyze_top_bottom_classes, plot_metrics_over_epochs, plot_loss_curves
from helpers.test import run_reproduction
from helpers.logger import setup_logger
from torch import inf, nn
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import numpy as np

# Setup logger
logger = setup_logger()

train_loss_array = []
test_loss_array = []
best_loss = inf
NUM_CLASSES = 10 

if __name__ == '__main__':
    logger.info("Deep learning project by Jonas, Robin and Lukas.")

    parser = argparse.ArgumentParser(description="Train and evaluate a neural network on EuroSAT dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file (optional).",
    )
    parser.add_argument("--seed", type=int, help="Seed for all RNGs (e.g. matriculation number).")
    parser.add_argument("--data-path", type=str, default=None, help="Path to dataset root (e.g. ./data/EuroSAT_RGB or ./data/EuroSAT_MS)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--workers", type=int, help="Number of DataLoader workers.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Weight decay for optimizer.")
    parser.add_argument(
        "--augmentation",
        type=str,
        nargs="+",
        choices=["none", "mild", "strong", "resnet"],
        help="List of augmentations/preprocessing tags: none, mild, strong, resnet.",
    )
    parser.add_argument(
        "--project-path",
        type=str,
        help="Base project path; data paths are resolved relative to this.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["rgb", "ms"],
        help="Image source to use (rgb or ms). Overrides config if set.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet"],
        help="Model type (cnn or resnet).",
    )
    parser.add_argument("--reproduction", action="store_true", help="Run in reproduction mode (skip training, load model, check logits).")
    parser.add_argument("--save-logits", action="store_true", help="Save computed logits (only in reproduction mode).")
    parser.add_argument("--logits-path", type=str, default="logits.pt", help="Path to the logits file.")
    parser.add_argument("--model-path", type=str, default="model.pth", help="Path to save/load the model.")

    args = parser.parse_args()

    # Konfiguration laden (Defaults -> optionale Datei -> CLI-Overrides)
    cfg = load_config(args)

    # Gewählte Konfiguration einmalig ausgeben
    logger.info("Effective configuration", extra={"extra_data": {"config": cfg}})

    SEED = int(cfg["seed"])
    set_fixed_seed(SEED)

    # Basis-Pfad für das Projekt; alle relativen Pfade werden hiervon aus aufgelöst
    project_path = os.path.abspath(cfg["project_path"])

    use_ms = cfg["data_source"] == "ms"

    # Gemeinsame Augmentierungs-Pipelines für RGB und MS
    augment_mild = [
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    augment_strong = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
    ]

    # ResNet-Preprocessing (nur für RGB + ResNet-Modelle sinnvoll)
    resnet_preprocess = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    # Datenpfad aus Config (mit Fallback auf Code-Defaults)
    def resolve_path(base: str, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    data_path = resolve_path(project_path, cfg["data_path"])

    # augmentation-Config als Liste von Tags behandeln (Config und CLI liefern Listen)
    aug_tags = cfg.get("augmentation", [])
    if not isinstance(aug_tags, (list, tuple)):
        aug_tags = [aug_tags]

    # Wenn "none" gesetzt ist, alle anderen Augmentations ignorieren
    if "none" in aug_tags:
        aug_tags = []
    
    # Im Reproduction-Mode keine zufälligen Augmentations verwenden
    if args.reproduction:
        aug_tags = [t for t in aug_tags if t == "resnet"]

    # Liste von Transforms für Training zusammenbauen
    transform_list = []
    if "mild" in aug_tags:
        transform_list.extend(augment_mild)
    if "strong" in aug_tags:
        transform_list.extend(augment_strong)
    
    model_kind = cfg["model"]  # "cnn" oder "resnet"
    if (not use_ms) and model_kind == "resnet" and "resnet" in aug_tags:
        transform_list.extend(resnet_preprocess)
        eval_transform = transforms.Compose(resnet_preprocess)
    else:
        if "resnet" in aug_tags:
            logger.warning("ResNet preprocessing requested but not applicable (only for RGB + ResNet). Ignoring..")
        eval_transform = transforms.Lambda(lambda x: x)

    # Falls keine Transforms gewählt wurden, Identität verwenden
    if transform_list:
        train_transform = transforms.Compose(transform_list)
    else:
        train_transform = transforms.Lambda(lambda x: x)

    dr = DataReader(data_path=data_path, eval_transform=eval_transform, train_transform=train_transform, use_ms=use_ms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Modellwahl abhängig von Datenquelle und gemeinsamem 'model'-Schalter
    if use_ms:
        if model_kind == "resnet":
            model = PretrainedResNet18MS().to(device)
        else:
            model = NeuralNetworkMS().to(device)
    else:
        if model_kind == "resnet":
            model = PretrainedResNet18().to(device)
        else:
            model = NeuralNetwork().to(device)

    # Reproduction Mode: Skip Training, Load Model, Run Test
    if args.reproduction:
        logger.info("Starting Reproduction Mode")
        model_path = args.model_path
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning(f"{model_path} not found. Using random weights (reproduction might fail).")
        
        idx_to_label = {idx: label for label, idx in dr.label_to_idx.items()}
        
        # Use test_dataloader for reproduction
        test_dataloader = DataLoader(dr.test_set, batch_size=int(cfg["batch_size"]), shuffle=False)
        
        run_reproduction(
            test_dataloader, 
            model, 
            device, 
            args.save_logits, 
            args.logits_path,
            idx_to_label
        )
        exit(0)

    loss_fn = nn.CrossEntropyLoss()
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    weight_decay = float(cfg["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    shuffling = True

    batch_size = int(cfg["batch_size"])
    workers = int(cfg["workers"])

    train_dataloader = DataLoader(dr.training_set, batch_size=batch_size, shuffle=shuffling, num_workers=workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(dr.validation_set, batch_size=batch_size, shuffle=shuffling)
    test_dataloader = DataLoader(dr.test_set, batch_size=batch_size, shuffle=shuffling)

        
    val_acc_epochs = []
    val_tpr_epochs = []

    for t in range(epochs):
        logger.info(f"Starting Epoch {t+1}")
        
        train(train_dataloader, model, loss_fn, optimizer, device, train_loss_array)
        acc, tpr, best_loss, _, _ = evaluate(
            val_dataloader,
            model,
            loss_fn,
            device,
            save_best=True,
            test_loss_array=test_loss_array,
            best_loss=best_loss,
            model_path=args.model_path,
        )
        val_acc_epochs.append(acc)
        val_tpr_epochs.append(tpr)
        
    logger.info("Training Done!")


    model.load_state_dict(torch.load(args.model_path, map_location=device))
    test_acc, test_tpr, _, test_logits, test_labels = evaluate(
        test_dataloader,
        model,
        loss_fn,
        device,
        save_best=False,
        test_loss_array=test_loss_array,
        best_loss=best_loss,
        model_path=args.model_path,
    )


    idx_to_label = {idx: label for label, idx in dr.label_to_idx.items()}
    analyze_top_bottom_classes(
        dr.test_set, 
        test_logits,
        test_labels,
        idx_to_label=idx_to_label, 
        class_indices=[0, 1, 2], 
        top_k=5, 
        bottom_k=5, 
        use_ms=use_ms
    )

    plot_metrics_over_epochs(epochs, val_acc_epochs, val_tpr_epochs, idx_to_label)
    plot_loss_curves(epochs, train_loss_array, test_loss_array)