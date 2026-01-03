import argparse
import os
import random
from helpers.dataset import DataReader
from helpers.nn import NeuralNetwork, NeuralNetworkMS, PretrainedResNet18, PretrainedResNet18MS
from helpers.config import load_config
from helpers.training import set_fixed_seed, train, evaluate
from helpers.plotting import analyze_top_bottom_classes, plot_metrics_over_epochs, plot_loss_curves
from torch import inf, nn
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import numpy as np

train_loss_array = []
test_loss_array = []
best_loss = inf
NUM_CLASSES = 10 

if __name__ == '__main__':
    print("Deep learning project by Jonas, Robin and Lukas.")

    parser = argparse.ArgumentParser(description="Train and evaluate a neural network on EuroSAT dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file (optional).",
    )
    parser.add_argument(
        "--use-ms",
        action="store_true",
        help="Use MS images instead of RGB.",
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
        "--model-rgb",
        type=str,
        choices=["cnn", "pretrained_resnet"],
        help="Model type for RGB data.",
    )
    # Alte Argumente beibehalten, aber intern auf gemeinsamen 'model'-Key mappen
    parser.add_argument(
        "--model-ms",
        type=str,
        choices=["cnn", "pretrained_resnet"],
        help="(legacy) Model type for MS data (mapped to --model).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "resnet"],
        help="Model type (cnn or resnet).",
    )
    args = parser.parse_args()

    # Konfiguration laden (Defaults -> optionale Datei -> CLI-Overrides)
    cfg = load_config(args)

    # Gewählte Konfiguration einmalig ausgeben
    print("Effective configuration:")
    for key in sorted(cfg.keys()):
        print(f"  {key}: {cfg[key]}")

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

    # Liste von Transforms für Training zusammenbauen
    transform_list = []
    if "mild" in aug_tags:
        transform_list.extend(augment_mild)
    if "strong" in aug_tags:
        transform_list.extend(augment_strong)

    # ResNet-Preprocessing nur für RGB + ResNet hinzufügen, wenn explizit gewünscht
    model_kind = cfg["model"]  # "cnn" oder "resnet"
    if (not use_ms) and model_kind == "resnet" and "resnet" in aug_tags:
        transform_list.extend(resnet_preprocess)

    # Falls keine Transforms gewählt wurden, Identität verwenden
    if transform_list:
        train_transform = transforms.Compose(transform_list)
    else:
        train_transform = transforms.Lambda(lambda x: x)

    dr = DataReader(data_path=data_path, transform=train_transform, use_ms=use_ms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You re using {device} :)")

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

    loss_fn = nn.CrossEntropyLoss()
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    weight_decay = float(cfg["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    shuffling = False

    batch_size = int(cfg["batch_size"])
    workers = int(cfg["workers"])

    train_dataloader = DataLoader(dr.training_set, batch_size=batch_size, shuffle=shuffling, num_workers=workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(dr.validation_set, batch_size=batch_size, shuffle=shuffling)
    test_dataloader = DataLoader(dr.test_set, batch_size=batch_size, shuffle=shuffling)

        
    val_acc_epochs = []
    val_tpr_epochs = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train(train_dataloader, model, loss_fn, optimizer, device, train_loss_array)
        acc, tpr, best_loss = evaluate(
            val_dataloader,
            model,
            loss_fn,
            device,
            save_best=True,
            test_loss_array=test_loss_array,
            best_loss=best_loss,
            model_path="model.pth",
        )
        val_acc_epochs.append(acc)
        val_tpr_epochs.append(tpr)
        
    print("Done!")


    model.load_state_dict(torch.load("model.pth", map_location=device))
    test_acc, test_tpr, _ = evaluate(
        test_dataloader,
        model,
        loss_fn,
        device,
        save_best=False,
        test_loss_array=test_loss_array,
        best_loss=best_loss,
        model_path="model.pth",
    )


    idx_to_label = {idx: label for label, idx in dr.label_to_idx.items()}
    analyze_top_bottom_classes(model, dr.test_set, device, idx_to_label=idx_to_label, class_indices=[0, 1, 2], top_k=5, bottom_k=5, use_ms=use_ms)

    plot_metrics_over_epochs(epochs, val_acc_epochs, val_tpr_epochs, idx_to_label)
    plot_loss_curves(epochs, train_loss_array, test_loss_array)