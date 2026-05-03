"""
Metrics collection and logging for training.

Tracks:
- throughput (tokens/sec)
- GPU utilization
- communication payload bytes
- GSM8K accuracy
- perplexity
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    global_step: int = 0
    timestamp: float = field(default_factory=time.time)

    # Training loss
    loss: float = 0.0
    loss_ema: float = 0.0

    # Throughput
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    batch_size: int = 0

    # Gradients
    grad_norm: float = 0.0
    grad_norm_ema: float = 0.0

    # Quantization
    compression_ratio: float = 1.0
    quantized_payload_bytes: int = 0

    # Communication
    comm_time_sec: float = 0.0
    param_pull_time_sec: float = 0.0
    grad_push_time_sec: float = 0.0

    # Distributed
    staleness: int = 0
    param_version: int = 0

    # Evaluation
    gsm8k_accuracy: Optional[float] = None
    math_accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    eval_step: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary (for logging)."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MetricsLogger:
    """
    Logs and aggregates training metrics.

    Provides:
    - exponential moving averages
    - periodic checkpointing
    - summary statistics
    """

    def __init__(self, ema_alpha: float = 0.1, log_interval: int = 10):
        self.ema_alpha = ema_alpha
        self.log_interval = log_interval
        self.metrics_history = []
        self.current_metrics = TrainingMetrics()

    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if hasattr(self.current_metrics, key):
                setattr(self.current_metrics, key, value)

        # Update EMA for loss and grad norm
        if 'loss' in kwargs:
            loss = kwargs['loss']
            if self.current_metrics.loss_ema == 0:
                self.current_metrics.loss_ema = loss
            else:
                self.current_metrics.loss_ema = (
                    self.ema_alpha * loss
                    + (1 - self.ema_alpha) * self.current_metrics.loss_ema
                )

        if 'grad_norm' in kwargs:
            grad_norm = kwargs['grad_norm']
            if self.current_metrics.grad_norm_ema == 0:
                self.current_metrics.grad_norm_ema = grad_norm
            else:
                self.current_metrics.grad_norm_ema = (
                    self.ema_alpha * grad_norm
                    + (1 - self.ema_alpha) * self.current_metrics.grad_norm_ema
                )

    def should_log(self) -> bool:
        """Check if should log at this step."""
        return self.current_metrics.global_step % self.log_interval == 0

    def log_step(self) -> TrainingMetrics:
        """Record current metrics and return copy."""
        metrics_copy = TrainingMetrics(**asdict(self.current_metrics))
        self.metrics_history.append(metrics_copy)
        return metrics_copy

    def get_summary(self) -> Dict:
        """Get summary statistics across all steps."""
        if not self.metrics_history:
            return {}

        losses = [m.loss for m in self.metrics_history]
        grad_norms = [m.grad_norm for m in self.metrics_history]
        throughputs = [m.tokens_per_sec for m in self.metrics_history if m.tokens_per_sec > 0]
        staleness = [m.staleness for m in self.metrics_history]

        return {
            'total_steps': len(self.metrics_history),
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'min_loss': min(losses) if losses else 0,
            'max_loss': max(losses) if losses else 0,
            'avg_grad_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
            'avg_throughput_tokens_per_sec': sum(throughputs) / len(throughputs) if throughputs else 0,
            'avg_staleness': sum(staleness) / len(staleness) if staleness else 0,
        }

    def save_logs(self, filepath: str):
        """Save metrics to JSON file."""
        logs = {
            'summary': self.get_summary(),
            'steps': [m.to_dict() for m in self.metrics_history]
        }
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2)

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        print("\n=== Training Metrics Summary ===")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        print("=" * 40 + "\n")


class EvaluationMetrics:
    """Metrics for evaluation on downstream tasks."""

    def __init__(self):
        self.gsm8k_results = {}  # step -> accuracy
        self.math_results = {}   # step -> accuracy
        self.perplexity_results = {}  # step -> perplexity

    def log_gsm8k(self, step: int, accuracy: float, num_correct: int, total: int):
        """Log GSM8K evaluation result."""
        self.gsm8k_results[step] = {
            'accuracy': accuracy,
            'correct': num_correct,
            'total': total
        }

    def log_math(self, step: int, accuracy: float, num_correct: int, total: int):
        """Log MATH subset evaluation result."""
        self.math_results[step] = {
            'accuracy': accuracy,
            'correct': num_correct,
            'total': total
        }

    def log_perplexity(self, step: int, perplexity: float, num_tokens: int):
        """Log perplexity on held-out set."""
        self.perplexity_results[step] = {
            'perplexity': perplexity,
            'num_tokens': num_tokens
        }

    def get_best_gsm8k(self) -> Optional[Dict]:
        """Get best GSM8K result."""
        if not self.gsm8k_results:
            return None
        best_step = max(self.gsm8k_results, key=lambda s: self.gsm8k_results[s]['accuracy'])
        return {
            'step': best_step,
            **self.gsm8k_results[best_step]
        }

    def get_best_math(self) -> Optional[Dict]:
        """Get best MATH result."""
        if not self.math_results:
            return None
        best_step = max(self.math_results, key=lambda s: self.math_results[s]['accuracy'])
        return {
            'step': best_step,
            **self.math_results[best_step]
        }

    def get_summary(self) -> Dict:
        """Get evaluation summary."""
        return {
            'gsm8k': self.get_best_gsm8k(),
            'math': self.get_best_math(),
            'num_gsm8k_evals': len(self.gsm8k_results),
            'num_math_evals': len(self.math_results),
            'num_perplexity_evals': len(self.perplexity_results)
        }

    def save_results(self, filepath: str):
        """Save evaluation results to JSON."""
        results = {
            'gsm8k': self.gsm8k_results,
            'math': self.math_results,
            'perplexity': self.perplexity_results,
            'summary': self.get_summary()
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


def test_metrics():
    """Test metrics logger."""
    logger = MetricsLogger(log_interval=5)

    # Simulate training steps
    for step in range(50):
        logger.current_metrics.global_step = step
        logger.update(
            loss=1.0 / (step + 1),
            grad_norm=0.1 * (1 + 0.01 * step),
            tokens_per_sec=1000 + 100 * step,
            batch_size=32,
            staleness=step % 5
        )

        if logger.should_log():
            metrics = logger.log_step()
            print(f"Step {step}: loss={metrics.loss_ema:.4f}, "
                  f"grad_norm={metrics.grad_norm_ema:.4f}, "
                  f"throughput={metrics.tokens_per_sec:.0f} tokens/sec")

    logger.print_summary()


if __name__ == "__main__":
    test_metrics()
