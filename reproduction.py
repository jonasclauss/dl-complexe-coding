import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from helpers.dataset import DataReader
from helpers.nn import NeuralNetwork


SEED = 3792567
DATA_PATH = "./data/EuroSAT_RGB"
BATCH_SIZE = 128
BASELINE_LOGITS_PATH = "test_logits_baseline.pt"


def set_fixed_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(device: torch.device) -> torch.nn.Module:
    model = NeuralNetwork().to(device)
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_test_loader(seed: int) -> DataLoader:
    dr = DataReader(data_path=DATA_PATH, seed=seed)
    test_dataset = dr.test_set
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader


def compute_test_logits(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    logits_list = []
    model.eval()

    with torch.inference_mode():
        for X, _ in dataloader:
            X = X.to(device)
            out = model(X)  # (batch_size, nclasses)
            logits_list.append(out.cpu())

    logits = torch.cat(logits_list, dim=0)
    return logits


def main(save_baseline: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_fixed_seed(SEED)

    model = load_model(device)
    test_loader = get_test_loader(seed=SEED)

    logits = compute_test_logits(model, test_loader, device)
    print(f"Current test logits shape: {tuple(logits.shape)}")

    if save_baseline:
        torch.save(logits, BASELINE_LOGITS_PATH)
        print(f"Saved baseline logits to '{BASELINE_LOGITS_PATH}'")
        return

    if not os.path.exists(BASELINE_LOGITS_PATH):
        raise FileNotFoundError(
            f"Baseline logits file '{BASELINE_LOGITS_PATH}' not found. "
            f"Run with --save-baseline once to create it."
        )

    baseline_logits = torch.load(BASELINE_LOGITS_PATH, map_location="cpu")
    print(f"Loaded baseline logits shape: {tuple(baseline_logits.shape)}")

    if baseline_logits.shape != logits.shape:
        print("Shape mismatch between current and baseline logits!")
        print(f"  baseline: {tuple(baseline_logits.shape)}")
        print(f"  current : {tuple(logits.shape)}")
        return

    equal = torch.allclose(logits, baseline_logits, rtol=1e-5, atol=1e-6)
    max_abs_diff = (logits - baseline_logits).abs().max().item()

    print(f"Allclose to baseline: {equal}")
    print(f"Max absolute difference: {max_abs_diff:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducibility check for test logits.")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current test logits as baseline instead of comparing.",
    )
    args = parser.parse_args()

    main(save_baseline=args.save_baseline)
