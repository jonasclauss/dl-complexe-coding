import argparse
import os
import random
from helpers.dataset import DataReader
from helpers.nn import NeuralNetwork, NeuralNetworkMS
from torch import inf, nn
from torch.utils.data import DataLoader
import torch
from alive_progress import alive_bar
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

train_loss_array = []
test_loss_array = []
best_loss = inf
NUM_CLASSES = 10 

def train(dataloader, model, loss_fn, optimizer):
    global train_loss_array
    size = len(dataloader)
    model.train()
    with alive_bar(size, title='Training', bar='bubbles') as bar:
        loss_arr = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred: torch.Tensor = model(X)
            y = y.long().to(device)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item()) 
            #loss, current = loss.item(), (batch + 1) * len(X)
            bar()
        train_loss_array.append(sum(loss_arr)/len(loss_arr))
        

def evaluate(dataloader, model, loss_fn, save_best = False):
    global best_loss, test_loss_array
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = y.long().to(device)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.append(pred.argmax(1).cpu())
            all_labels.append(y.cpu())
            
    test_loss /= num_batches
    correct /= size

    labels_cat = torch.cat(all_labels)
    preds_cat = torch.cat(all_preds)
    num_classes = max(len(torch.unique(labels_cat)), len(torch.unique(preds_cat)))
    conf = build_confusion_matrix(labels_cat, preds_cat, num_classes)

    # TPR pro Klasse (Recall)
    tp = conf.diag()
    per_class_total = conf.sum(dim=1).clamp(min=1)
    tpr_per_class = tp.float() / per_class_total.float()

    print(conf)
    if save_best:
        test_loss_array.append(test_loss)

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("TPR per class:", tpr_per_class.tolist())

    if save_best and best_loss > test_loss:
        torch.save(model.state_dict(), 'model.pth')
        best_loss = test_loss

    return correct, tpr_per_class.cpu()



def build_confusion_matrix(labels: torch.Tensor,
                           preds: torch.Tensor,
                           num_classes: int) -> torch.Tensor:
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    return conf

def set_fixed_seed(matno):
    torch.manual_seed(matno)
    torch.cuda.manual_seed_all(matno)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(matno)
    random.seed(matno)


def get_top_bottom_for_class(model, dataset, device, class_idx: int, top_k: int = 5, bottom_k: int = 5):
    """Berechne für eine Klasse die Top-k und Bottom-k Beispiele nach Modell-Score.

    Score basiert auf der CrossEntropy-Loss für class_idx.
    Liefert Listen von (loss, sample_index, predicted_class).
    """
    model.eval()
    scores = []
    loss_fn = nn.CrossEntropyLoss(reduction='none') 

    with torch.inference_mode():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            if int(label) != class_idx:
                continue

            x = image.unsqueeze(0).to(device)
            y = torch.tensor([class_idx], device=device)
            out = model(x)
            loss = loss_fn(out, y).item()
            pred_class = out.argmax(1).item()
            scores.append((loss, idx, pred_class))

    if not scores:
        return [], []

    # Nach Loss aufsteigend sortieren: kleine Loss = "Top" (leichte Beispiele), große Loss = "Bottom" (schwere Beispiele)
    scores_sorted = sorted(scores, key=lambda t: t[0])
    top = scores_sorted[:top_k]
    bottom = scores_sorted[-bottom_k:] if len(scores_sorted) >= bottom_k else scores_sorted
    return top, bottom


def plot_top_bottom_for_class(dataset, class_idx: int, top_examples, bottom_examples, idx_to_label, out_path: str, use_ms: bool = False):
    """Speichere ein Bild mit Top- und Bottom-Beispielen für eine Klasse.

    top_examples / bottom_examples: Listen von (loss, sample_index, predicted_class).
    """
    num_top = len(top_examples)
    num_bottom = len(bottom_examples)
    max_k = max(num_top, num_bottom)

    if max_k == 0:
        return

    fig, axes = plt.subplots(2, max_k, figsize=(3 * max_k, 6))

    # Falls nur ein Spalten-Subplot, axes in 2D-Form bringen
    if max_k == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    class_name = idx_to_label.get(class_idx, str(class_idx)) if idx_to_label is not None else str(class_idx)

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
                red   = image[3, ...]  # B04 - Red
                green = image[2, ...]  # B03 - Green
                blue  = image[1, ...]  # B02 - Blue
                vis = torch.stack([red, green, blue], dim=0)

                # Per-Bild min-max Normalisierung für sichtbaren Kontrast
                vis = vis.clone()
                vmin = float(vis.min())
                vmax = float(vis.max())
                if vmax > vmin:
                    vis = (vis - vmin) / (vmax - vmin)
                img_np = vis.permute(1, 2, 0).numpy()
                ax.imshow(img_np)
            else:
                # RGB: alle 3 Kanäle verwenden (hier schon sinnvoll skaliert)
                img_np = image.permute(1, 2, 0).numpy()
                ax.imshow(img_np)
            pred_name = idx_to_label.get(pred_class, str(pred_class)) if idx_to_label is not None else str(pred_class)
            ax.set_title(f"{row_title} {col+1}\nloss={loss:.2f}\npred={pred_name} ({pred_class})", fontsize=8)

    fig.suptitle(f"Class {class_idx} ({class_name}) - Top/Bottom Beispiele", fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(out_path)
    plt.close(fig)


def analyze_top_bottom_classes(model, test_dataset, device, idx_to_label=None, class_indices=None, top_k: int = 5, bottom_k: int = 5, use_ms: bool = False):
    """Für mehrere Klassen Top-5 und Bottom-5 Beispiele finden und plotten."""
    if class_indices is None:
        class_indices = [0, 1, 2]

    for class_idx in class_indices:
        top, bottom = get_top_bottom_for_class(model, test_dataset, device, class_idx, top_k=top_k, bottom_k=bottom_k)
        if not top and not bottom:
            continue
        class_name = idx_to_label.get(class_idx, str(class_idx)) if idx_to_label is not None else str(class_idx)
        filename = f"ranking_class_{class_idx}_{class_name}.png"
        plot_top_bottom_for_class(test_dataset, class_idx, top, bottom, idx_to_label, filename, use_ms=use_ms)

if __name__ == '__main__':
    print("Deep learning project by Jonas, Robin and Lukas.")

    parser = argparse.ArgumentParser(description="Train and evaluate a neural network on EuroSAT dataset.")
    parser.add_argument(
        "--use-ms",
        action="store_true",
        help="Use MS images instead of RGB.",
    )
    args = parser.parse_args()

    MATRIKENUMMER = 3792567
    set_fixed_seed(MATRIKENUMMER)

    train_transform_mild = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5)
    ])

    train_transform_strong = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
    # transforms.ColorJitter(
    #     brightness=0.2,
    #     contrast=0.2,
    #     saturation=0.2,
    #     hue=0.05
    # ),
    ])


    dr = DataReader(data_path= './data/EuroSAT_MS' if args.use_ms else './data/EuroSAT_RGB', transform=train_transform_mild, use_ms=args.use_ms)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"You re using {device} :)")

    if args.use_ms:
        model = NeuralNetworkMS().to(device)
    else:
        model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    shuffling = False

    batch_size = 128
    workers = 6
    # Create data loaders.
    train_dataloader = DataLoader(dr.training_set, batch_size=batch_size, shuffle=shuffling, num_workers=workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(dr.validation_set, batch_size=batch_size, shuffle=shuffling)
    test_dataloader = DataLoader(dr.test_set, batch_size=batch_size, shuffle=shuffling)

        
    val_acc_epochs = []
    val_tpr_epochs = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train(train_dataloader, model, loss_fn, optimizer)
        acc, tpr = evaluate(val_dataloader, model, loss_fn, save_best=True)
        val_acc_epochs.append(acc)
        val_tpr_epochs.append(tpr)
        
    print("Done!")

    #testing stuff
    if args.use_ms:
        model_test = NeuralNetworkMS().to(device)
    else:
        model_test = NeuralNetwork().to(device)
    model_test.load_state_dict(torch.load('model.pth'))
    test_acc, test_tpr = evaluate(test_dataloader, model_test, loss_fn, save_best=False)

    # Ranking-Analyse: für 3 Klassen Top-5 / Bottom-5 Testbilder plotten
    # baue Mapping von Klassenindex zu Label-Name
    idx_to_label = {idx: label for label, idx in dr.label_to_idx.items()}
    analyze_top_bottom_classes(model_test, dr.test_set, device, idx_to_label=idx_to_label, class_indices=[0, 1, 2], top_k=5, bottom_k=5, use_ms=args.use_ms)

    # Gemeinsamer Plot: Validation Accuracy und TPR pro Klasse (Werte 0 bis 1)
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
    plt.savefig("val_acc_tpr_per_class.png")
    plt.close()

    # Unveränderter Loss-Plot (Training/Test Loss über Epochen)
    plt.figure()
    plt.plot(range(1, epochs+1), train_loss_array, label='Training Loss')
    plt.plot(range(1, epochs+1), test_loss_array, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()