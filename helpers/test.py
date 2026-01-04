import os
import torch
from torch.utils.data import DataLoader
from helpers.training import evaluate
from helpers.logger import setup_logger

logger = setup_logger()

def run_reproduction(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    save_baseline: bool,
    baseline_path: str,
    idx_to_label: dict = None
) -> None:
    """
    Runs the reproduction routine: computes logits using evaluate(), prints predictions,
    and either saves logits to baseline_path or compares them with existing baseline.
    """
    logger.info("Running reproduction (inference)...")
    
    # Dummy loss function and variables for evaluate
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Use evaluate to get logits
    _, _, _, logits, _ = evaluate(
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        device=device,
        save_best=False,
        test_loss_array=[],
        best_loss=float('inf'),
        model_path=""
    )

    # Print sample predictions
    probs = torch.softmax(logits, dim=1)
    confs, preds = torch.max(probs, dim=1)
    limit = min(len(preds), 10)
    
    sample_predictions = []
    for j in range(limit):
        pred_idx = preds[j].item()
        label_name = idx_to_label[pred_idx] if idx_to_label else str(pred_idx)
        sample_predictions.append({
            "sample_idx": j,
            "class_idx": pred_idx,
            "label": label_name,
            "confidence": f"{confs[j].item():.4f}"
        })
    
    logger.info("Sample Predictions (First 10)", extra={"extra_data": {"samples": sample_predictions}})
    
    if save_baseline:
        torch.save(logits, baseline_path)
        logger.info(f"Saved baseline logits to '{baseline_path}'")
    else:
        if not os.path.exists(baseline_path):
            logger.error(f"Baseline file '{baseline_path}' not found.")
            logger.info("Run with --save-baseline to create a new baseline file.")
            return

        baseline_logits = torch.load(baseline_path, map_location="cpu")
        
        if baseline_logits.shape != logits.shape:
            logger.error(f"Shape mismatch! Baseline: {tuple(baseline_logits.shape)}, Current: {tuple(logits.shape)}")
            return

        equal = torch.allclose(logits, baseline_logits, rtol=1e-5, atol=1e-6)
        max_abs_diff = (logits - baseline_logits).abs().max().item()

        logger.info("Baseline Comparison", extra={"extra_data": {
            "all_close": equal,
            "max_abs_diff": f"{max_abs_diff:.3e}"
        }})
        
        if equal:
            logger.info("SUCCESS: Reproduction check passed.")
        else:
            logger.error("FAILURE: Reproduction check failed.")
