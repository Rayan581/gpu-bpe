"""
Full system training: GPU tokenizer + HSG + adaptive compression + DC-ASGD.

Integrates:
- GPU-aware byte-level BPE tokenization
- Hybrid Semantic Guard (digit-span locking)
- Adaptive layer-wise gradient quantization (INT8)
- Three-tier distributed architecture with DC-ASGD
- Error feedback compensation

Usage:
    python train_full.py --num_steps 100 --batch_size 32 --num_workers 2
"""

import torch
import torch.nn as nn
import sys
import argparse
import time
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizer.gpu_bpe import GPUBPETokenizer
from tokenizer.hsg import SemanticGuardedTokenizer
from compression.adaptive_quant import AdaptiveQuantizer
from dist.control import ControlLayer
from dist.parameter_server import ParameterServer
from dist.worker import TrainingWorker
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


class DistributedTrainer:
    """Manages distributed training with multiple workers."""

    def __init__(
        self,
        num_workers: int = 2,
        max_staleness: int = 5,
        device: torch.device = None
    ):
        self.num_workers = num_workers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared components
        self.model = create_model()
        self.param_server = ParameterServer(self.model, device=self.device)
        self.control = ControlLayer(num_workers, max_staleness=max_staleness)
        self.quantizer = AdaptiveQuantizer(
            self.model,
            bits=8,
            update_interval=50,
            device=self.device
        )

        # Workers
        self.workers = []
        for worker_id in range(num_workers):
            worker_model = create_model()

            def compress_fn(grads):
                return self.quantizer.quantize_grads(grads)

            worker = TrainingWorker(
                worker_id=worker_id,
                model=worker_model,
                parameter_server=self.param_server,
                tokenizer=None,
                device=self.device,
                compression_fn=compress_fn
            )
            self.workers.append(worker)

    def train_step(
        self,
        batches: List,
        criterion: nn.Module
    ):
        """Execute one distributed training step."""
        # Assign batches to workers
        for worker_id, batch in enumerate(batches[:self.num_workers]):
            worker = self.workers[worker_id]
            worker.step(batch, criterion, pull_params=True, push_grads=True)

        # Aggregate gradients (synchronization point)
        if self.control.need_sync():
            aggregated = self.param_server.aggregate_gradients(
                self.control.global_step,
                self.num_workers
            )
            self.param_server.apply_gradients(aggregated, learning_rate=1e-4)
            sync_info = self.control.synchronize()
            return sync_info

        return None

    def get_metrics(self):
        """Collect metrics from all workers."""
        metrics = {
            'control_state': self.control.get_control_state(),
            'server_state': self.param_server.get_server_state(),
            'worker_states': [w.get_worker_state() for w in self.workers],
            'quantizer_stats': self.quantizer.get_stats()
        }
        return metrics


def train_full(
    num_steps: int = 100,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 2,
    max_staleness: int = 5,
    enable_hsg: bool = True,
    save_dir: str = "./outputs/full",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run full system training.

    Args:
        num_steps: total training steps
        batch_size: batch size
        max_length: max sequence length
        num_workers: number of distributed workers
        max_staleness: max staleness for DC-ASGD
        enable_hsg: whether to enable Hybrid Semantic Guard
        save_dir: where to save logs
        device: compute device
    """
    print(f"=== Full System Training ===")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Max staleness: {max_staleness}")
    print(f"HSG enabled: {enable_hsg}")
    print(f"Num steps: {num_steps}\n")

    device = torch.device(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Create tokenizer (GPU-aware with HSG)
    print("Creating tokenizer...")
    base_tokenizer = GPUBPETokenizer(vocab_size=50257, use_gpu=True)
    tokenizer = SemanticGuardedTokenizer(base_tokenizer, enable_hsg=enable_hsg)

    # Create distributed trainer
    print("Creating distributed trainer...")
    trainer = DistributedTrainer(
        num_workers=num_workers,
        max_staleness=max_staleness,
        device=device
    )

    # Create data loader
    print("Creating data loader...")
    loader = get_dataloader(
        'owt',
        tokenizer,
        batch_size=batch_size * num_workers,
        num_docs=1000,
        max_length=max_length
    )

    criterion = nn.CrossEntropyLoss()
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

        # Create worker batches (distributed)
        if isinstance(batch['input_ids'], torch.Tensor):
            total_batch_size = batch['input_ids'].shape[0]
            worker_batch_size = total_batch_size // trainer.num_workers

            worker_batches = []
            for worker_id in range(trainer.num_workers):
                start_idx = worker_id * worker_batch_size
                end_idx = (worker_id + 1) * worker_batch_size

                if worker_id == trainer.num_workers - 1:
                    end_idx = total_batch_size

                if start_idx < end_idx:
                    worker_batch = (
                        batch['input_ids'][start_idx:end_idx],
                        batch['labels'][start_idx:end_idx]
                    )
                    worker_batches.append(worker_batch)

            # Distributed step
            sync_info = trainer.train_step(worker_batches, criterion)

            # Update metrics
            num_tokens = (batch['labels'] != 50256).sum().item()
            total_tokens += num_tokens

            # Get first worker's loss for logging
            worker_loss = trainer.workers[0].loss_history[-1] if trainer.workers[0].loss_history else 0.0

            metrics_logger.current_metrics.global_step = step
            metrics_logger.update(
                loss=worker_loss,
                grad_norm=0.1,  # Placeholder
                tokens_per_sec=num_tokens / (time.time() - step_start_time),
                batch_size=batch_size,
                staleness=trainer.control.staleness_tracker.get_worker_stats()[0]['staleness']
                         if trainer.control.staleness_tracker.worker_status else 0,
                compression_ratio=trainer.quantizer.get_compression_ratio(),
                quantized_payload_bytes=int(num_tokens * 32 / trainer.quantizer.get_compression_ratio() / 8)
            )

            if metrics_logger.should_log():
                metrics = metrics_logger.log_step()
                print(
                    f"Step {step}: "
                    f"loss={metrics.loss_ema:.4f} "
                    f"throughput={metrics.tokens_per_sec:.0f} tokens/sec "
                    f"compression={metrics.compression_ratio:.1f}x"
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
    parser = argparse.ArgumentParser(description="Full system training")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_staleness", type=int, default=5)
    parser.add_argument("--enable_hsg", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./outputs/full")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_full(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        max_staleness=args.max_staleness,
        enable_hsg=args.enable_hsg,
        save_dir=args.save_dir,
        device=device
    )
