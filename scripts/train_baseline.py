"""
Baseline training: CPU tokenizer, synchronous FP32 training.

No distributed overhead, no quantization, no HSG.
Serves as reference for measuring speedup.

Usage:
    python train_baseline.py --num_steps 100 --batch_size 32
"""

import torch
import torch.nn as nn
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizer.gpu_bpe import GPUBPETokenizer
from dist.worker import LocalTrainer
from utils.metrics import MetricsLogger
from utils.data import get_dataloader


def create_model(vocab_size: int = 50257, hidden_size: int = 768, num_layers: int = 12):
    """Create GPT-2 Small model."""
    return nn.Sequential(
        nn.Embedding(vocab_size, hidden_size),
        *[
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ],
        nn.Linear(hidden_size, vocab_size)
    )


def train_baseline(
    num_steps: int = 100,
    batch_size: int = 32,
    max_length: int = 512,
    num_eval_steps: int = 10,
    save_dir: str = "./outputs/baseline",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run baseline training.

    Args:
        num_steps: total training steps
        batch_size: batch size
        max_length: max sequence length
        num_eval_steps: evaluate every N steps
        save_dir: where to save logs
        device: compute device
    """
    print(f"=== Baseline Training ===")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print(f"Num steps: {num_steps}\n")

    # Setup
    device = torch.device(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = GPUBPETokenizer(vocab_size=50257, use_gpu=False)  # CPU-only

    # Create model
    print("Creating model...")
    model = create_model()
    model = model.to(device)

    # Create trainer
    trainer = LocalTrainer(model, tokenizer, device=device, use_compression=False)

    # Create data loader
    print("Creating data loader...")
    loader = get_dataloader(
        'owt',
        tokenizer,
        batch_size=batch_size,
        num_docs=1000,
        max_length=max_length
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Metrics
    metrics_logger = MetricsLogger(log_interval=max(1, num_steps // 20))

    # Training loop
    print("\nStarting training...")
    step = 0
    batch_iter = iter(loader)
    total_tokens = 0
    step_start_time = time.time()

    while step < num_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward/backward
        loss, grad_norm = trainer.train_step((input_ids, labels), criterion)

        # Update metrics
        num_tokens = (labels != 50256).sum().item()
        total_tokens += num_tokens

        metrics_logger.current_metrics.global_step = step
        metrics_logger.update(
            loss=loss,
            grad_norm=grad_norm,
            tokens_per_sec=num_tokens / (time.time() - step_start_time),
            batch_size=batch_size,
            staleness=0
        )

        if metrics_logger.should_log():
            metrics = metrics_logger.log_step()
            print(
                f"Step {step}: "
                f"loss={metrics.loss_ema:.4f} "
                f"grad_norm={metrics.grad_norm_ema:.4f} "
                f"throughput={metrics.tokens_per_sec:.0f} tokens/sec"
            )

        step += 1
        step_start_time = time.time()

    # Summary
    print("\n" + "=" * 50)
    metrics_logger.print_summary()

    # Save logs
    log_file = f"{save_dir}/metrics.json"
    metrics_logger.save_logs(log_file)
    print(f"Logs saved to {log_file}\n")

    return trainer, metrics_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline training")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="./outputs/baseline")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_baseline(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_dir=args.save_dir,
        device=device
    )
