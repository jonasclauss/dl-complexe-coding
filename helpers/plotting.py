from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


def get_top_bottom_for_class(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    class_idx: int,
    top_k: int = 5,
    bottom_k: int = 5,
) -> Tuple[List[Tuple[float, int, int]], List[Tuple[float, int, int]]]:
    """Berechne für eine Klasse die Top-k und Bottom-k Beispiele nach Modell-Score.

    Score basiert auf der CrossEntropy-Loss für die wahre Klasse.
    Liefert Listen von (loss, sample_index, predicted_class).
    """
    model.eval()
    scores: List[Tuple[float, int, int]] = []
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    with torch.inference_mode():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            if int(label) != class_idx:
                continue
            x = image.unsqueeze(0).to(device)
            y = torch.tensor([int(label)], device=device)
            out = model(x)
            loss = float(loss_fn(out, y)[0].item())
            pred_class = int(out.argmax(1).item())
            scores.append((loss, idx, pred_class))

    if not scores:
        return [], []

    # Nach Loss aufsteigend sortieren: kleine Loss = "Top" (leichte Beispiele),
    # große Loss = "Bottom" (schwere Beispiele)
    scores_sorted = sorted(scores, key=lambda t: t[0])
    top = scores_sorted[:top_k]
    bottom = scores_sorted[-bottom_k:] if len(scores_sorted) >= bottom_k else scores_sorted
    return top, bottom


def plot_top_bottom_for_class(
    dataset,
    class_idx: int,
    top_examples: Iterable[Tuple[float, int, int]],
    bottom_examples: Iterable[Tuple[float, int, int]],
    idx_to_label: Optional[dict],
    out_path: str,
    use_ms: bool = False,
) -> None:
    """Speichere ein Bild mit Top- und Bottom-Beispielen für eine Klasse.

    top_examples / bottom_examples: Listen von (loss, sample_index, predicted_class).
    """
    top_examples = list(top_examples)
    bottom_examples = list(bottom_examples)

    num_top = len(top_examples)
    num_bottom = len(bottom_examples)
    max_k = max(num_top, num_bottom)

    if max_k == 0:
        return

    fig, axes = plt.subplots(2, max_k, figsize=(3 * max_k, 6))

    # Falls nur ein Spalten-Subplot, axes in 2D-Form bringen
    if max_k == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    class_name = (
        idx_to_label.get(class_idx, str(class_idx)) if idx_to_label is not None else str(class_idx)
    )

    for row, examples, row_title in [(0, top_examples, "Top"), (1, bottom_examples, "Bottom")]:
        for col in range(max_k):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(examples):
                continue
            loss, sample_idx, pred_class = examples[col]
            image, _ = dataset[sample_idx]

            # Bild für Visualisierung vorbereiten
            # Tensor kommt als C x H x W, Werte bereits in [0,1]
            if use_ms:
                red = image[3, ...]  # B04 - Red
                green = image[2, ...]  # B03 - Green
                blue = image[1, ...]  # B02 - Blue
                vis = torch.stack([red, green, blue], dim=0)

                # Per-Bild min-max Normalisierung für sichtbaren Kontrast (MS)
                vis = vis.clone()
            else:
                # RGB: alle 3 Kanäle verwenden, vorher lokal auf [0,1] skalieren
                vis = image.float().clone()

            vmin = float(vis.min())
            vmax = float(vis.max())
            if vmax > vmin:
                vis = (vis - vmin) / (vmax - vmin)
            img_np = vis.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np)

            pred_name = (
                idx_to_label.get(pred_class, str(pred_class))
                if idx_to_label is not None
                else str(pred_class)
            )
            ax.set_title(
                f"{row_title} {col+1}\nloss={loss:.2f}\npred={pred_name} ({pred_class})",
                fontsize=8,
            )

    fig.suptitle(f"Class {class_idx} ({class_name}) - Top/Bottom Beispiele", fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(out_path)
    plt.close(fig)


def analyze_top_bottom_classes(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    idx_to_label: Optional[dict] = None,
    class_indices: Optional[Iterable[int]] = None,
    top_k: int = 5,
    bottom_k: int = 5,
    use_ms: bool = False,
) -> None:
    """Für mehrere Klassen Top- und Bottom-Beispiele finden und plotten."""
    if class_indices is None:
        class_indices = [0, 1, 2]

    for class_idx in class_indices:
        top, bottom = get_top_bottom_for_class(
            model, test_dataset, device, class_idx, top_k=top_k, bottom_k=bottom_k
        )
        if not top and not bottom:
            continue
        class_name = (
            idx_to_label.get(class_idx, str(class_idx))
            if idx_to_label is not None
            else str(class_idx)
        )
        filename = f"ranking_class_{class_idx}_{class_name}.png"
        plot_top_bottom_for_class(
            test_dataset,
            class_idx,
            top,
            bottom,
            idx_to_label,
            filename,
            use_ms=use_ms,
        )


def plot_metrics_over_epochs(
    epochs: int,
    val_acc_epochs,
    val_tpr_epochs,
    idx_to_label: dict,
    out_path: str = "val_acc_tpr_per_class.png",
) -> None:
    epochs_range = range(1, epochs + 1)
    plt.figure()
    plt.plot(epochs_range, [float(a) for a in val_acc_epochs], label="Val Accuracy")
    for cls_idx in range(len(val_tpr_epochs[0])):
        tprs_cls = [float(t[cls_idx]) for t in val_tpr_epochs]
        cls_name = idx_to_label.get(cls_idx, str(cls_idx))
        plt.plot(epochs_range, tprs_cls, label=cls_name)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value (0-1)")
    plt.title("Validation Accuracy and TPR per Class over Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss_curves(
    epochs: int,
    train_loss_history,
    test_loss_history,
    out_path: str = "loss_plot.png",
) -> None:
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss_history, label="Training Loss")
    plt.plot(range(1, epochs + 1), test_loss_history, label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss over Epochs")
    plt.legend()
    plt.savefig(out_path)
    plt.close()
