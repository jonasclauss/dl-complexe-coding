import random
from typing import List, Tuple

import numpy as np
import torch
from alive_progress import alive_bar
from torch import nn


def set_fixed_seed(seed: int) -> None:
    """Setze alle relevanten Zufalls-Seeds für Reproduzierbarkeit."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def build_confusion_matrix(labels: torch.Tensor,
                           preds: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
    """Einfache Confusion-Matrix (Zeile = True-Label, Spalte = Prediction)."""
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    return conf


def train(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loss_array: List[float],
) -> None:
    """Eine Trainings-Epoche ausführen und den durchschnittlichen Loss loggen."""
    size = len(dataloader)
    model.train()
    with alive_bar(size, title="Training", bar="bubbles") as bar:
        loss_arr = []
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred: torch.Tensor = model(X)
            y = y.long().to(device)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item())
            bar()
        if loss_arr:
            train_loss_array.append(sum(loss_arr) / len(loss_arr))


def evaluate(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    save_best: bool,
    test_loss_array: List[float],
    best_loss: float,
    model_path: str = "model.pth",
) -> Tuple[float, torch.Tensor, float]:
    """Evaluation auf einem Dataloader.

    Gibt (Accuracy, TPR-pro-Klasse, aktualisierter_best_loss) zurück und
    speichert optional das beste Modell.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.long().to(device)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.append(pred.argmax(1).cpu())
            all_labels.append(y.cpu())

    test_loss /= max(num_batches, 1)
    correct /= max(size, 1)

    labels_cat = torch.cat(all_labels) if all_labels else torch.tensor([])
    preds_cat = torch.cat(all_preds) if all_preds else torch.tensor([])
    num_classes = max(
        len(torch.unique(labels_cat)) if labels_cat.numel() > 0 else 0,
        len(torch.unique(preds_cat)) if preds_cat.numel() > 0 else 0,
    ) or 1
    conf = build_confusion_matrix(labels_cat, preds_cat, num_classes)

    # TPR pro Klasse (Recall)
    tp = conf.diag()
    per_class_total = conf.sum(dim=1).clamp(min=1)
    tpr_per_class = tp.float() / per_class_total.float()

    print(conf)
    if save_best:
        test_loss_array.append(test_loss)

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    print("TPR per class:", tpr_per_class.tolist())

    if save_best and best_loss > test_loss:
        torch.save(model.state_dict(), model_path)
        best_loss = test_loss

    return correct, tpr_per_class.cpu(), best_loss
