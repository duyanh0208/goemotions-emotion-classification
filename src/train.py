"""
============================================================
Training Script — Fine-tune BERT/RoBERTa on GoEmotions
============================================================

Usage:
    python -m src.train --config configs/bert_base_debug.yaml
    python -m src.train --config configs/bert_base.yaml
    python -m src.train --config configs/roberta_base.yaml

Override config từ command line:
    python -m src.train --config configs/bert_base.yaml --epochs 5
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import (
    NUM_LABELS,
    compute_pos_weights,
    create_dataloaders,
    load_goemotions,
)
from .evaluate import compute_metrics, evaluate_model, per_class_f1
from .models import EmotionClassifier
from .utils import (
    count_parameters,
    format_time,
    get_device,
    load_config,
    save_json,
    set_seed,
)


# ============================================================
# Argument Parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Tắt W&B logging (override config)",
    )
    # Override args (optional)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, device,
    epoch, max_grad_norm, log_every_n_steps, use_wandb,
):
    """Train 1 epoch, return avg loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Logging
        if (batch_idx + 1) % log_every_n_steps == 0:
            avg = total_loss / (batch_idx + 1)
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch} | Batch {batch_idx+1:>5}/{n_batches} | "
                f"Loss: {avg:.4f} | LR: {lr:.2e}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/lr": lr,
                    "step": epoch * n_batches + batch_idx,
                })

    return total_loss / n_batches


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.no_wandb:
        config["logging"]["use_wandb"] = False

    # Setup
    set_seed(config["training"]["seed"])
    device = get_device()

    # Print config
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    import yaml
    print(yaml.dump(config, indent=2, default_flow_style=False, sort_keys=False))

    # ============================================================
    # W&B Init
    # ============================================================
    use_wandb = config["logging"].get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config["logging"]["wandb_project"],
            name=config["experiment"]["name"],
            tags=config["experiment"].get("tags", []),
            config=config,
        )

    # ============================================================
    # Data
    # ============================================================
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    ds = load_goemotions(
        dataset_name=config["data"]["dataset_name"],
        config=config["data"]["config"],
        debug=config["data"].get("debug", False),
        debug_train_size=config["data"].get("debug_train_size", 1000),
        debug_val_size=config["data"].get("debug_val_size", 500),
    )
    print(f"Train: {len(ds['train']):,} | Val: {len(ds['validation']):,} | Test: {len(ds['test']):,}")

    print(f"\nLoading tokenizer: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    train_loader, val_loader, test_loader = create_dataloaders(
        ds, tokenizer,
        max_length=config["data"]["max_length"],
        train_batch_size=config["training"]["batch_size"],
        eval_batch_size=config["training"]["eval_batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # ============================================================
    # Model
    # ============================================================
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = EmotionClassifier(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    ).to(device)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params['trainable']:,}")
    print(f"Total parameters:     {n_params['total']:,}")

    # ============================================================
    # Loss with pos_weight
    # ============================================================
    if config["loss"]["use_pos_weight"]:
        print("\nComputing pos_weights cho class imbalance...")
        clip_min, clip_max = config["loss"]["pos_weight_clip"]
        pos_weights = compute_pos_weights(
            ds["train"],
            num_labels=NUM_LABELS,
            clip_min=clip_min,
            clip_max=clip_max,
        ).to(device)
        print(f"pos_weight range: [{pos_weights.min():.2f}, {pos_weights.max():.2f}]")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ============================================================
    # Optimizer & Scheduler
    # ============================================================
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["epochs"]
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTotal training steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")

    # ============================================================
    # Training Loop
    # ============================================================
    print("\n" + "=" * 60)
    print(f"BẮT ĐẦU TRAINING — {config['training']['epochs']} epochs")
    print("=" * 60)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1_macro = 0.0
    history = []
    start_time = time.time()

    for epoch in range(config["training"]["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            epoch=epoch + 1,
            max_grad_norm=config["training"]["max_grad_norm"],
            log_every_n_steps=config["logging"].get("log_every_n_steps", 50),
            use_wandb=use_wandb,
        )

        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss:   {train_loss:.4f}")
        print(f"  Val Loss:     {val_metrics['loss']:.4f}")
        print(f"  Val F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Val F1-micro: {val_metrics['f1_micro']:.4f}")
        print(f"  Hamming Loss: {val_metrics['hamming_loss']:.4f}")

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_epoch": train_loss,
                "val/loss": val_metrics["loss"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/f1_micro": val_metrics["f1_micro"],
                "val/hamming_loss": val_metrics["hamming_loss"],
            })

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        })

        # Save best model
        if val_metrics["f1_macro"] > best_f1_macro:
            best_f1_macro = val_metrics["f1_macro"]
            best_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, best_path)
            print(f"  ✅ Saved best model (F1-macro: {best_f1_macro:.4f})")

    total_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {format_time(total_time)}")

    # ============================================================
    # Final Test Evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_logits, test_labels = evaluate_model(model, test_loader, criterion, device)

    print(f"Test F1-macro:    {test_metrics['f1_macro']:.4f}")
    print(f"Test F1-micro:    {test_metrics['f1_micro']:.4f}")
    print(f"Test F1-weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")

    # Per-class F1
    class_f1 = per_class_f1(test_logits, test_labels)
    print("\n🔍 Per-class F1 (top 5 best / bottom 5):")
    sorted_classes = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)
    print("  Best:")
    for name, f1 in sorted_classes[:5]:
        print(f"    {name:<20} {f1:.4f}")
    print("  Worst:")
    for name, f1 in sorted_classes[-5:]:
        print(f"    {name:<20} {f1:.4f}")

    # Save results
    results = {
        "experiment": config["experiment"]["name"],
        "model_name": config["model"]["name"],
        "config": config,
        "history": history,
        "test_metrics": test_metrics,
        "test_per_class_f1": class_f1,
        "best_val_f1_macro": best_f1_macro,
        "total_training_time_seconds": total_time,
    }

    results_path = Path(config["paths"]["results_dir"]) / f"{config['experiment']['name']}.json"
    save_json(results, results_path)
    print(f"\n✅ Saved results: {results_path}")

    # Save predictions
    import numpy as np
    np.save(output_dir / "test_logits.npy", test_logits.numpy())
    np.save(output_dir / "test_labels.npy", test_labels.numpy())

    if use_wandb:
        import wandb
        wandb.log({
            "test/f1_macro": test_metrics["f1_macro"],
            "test/f1_micro": test_metrics["f1_micro"],
            "test/hamming_loss": test_metrics["hamming_loss"],
        })
        wandb.finish()

    print(f"\n🎉 Done! Best Val F1-macro: {best_f1_macro:.4f}")


if __name__ == "__main__":
    main()
