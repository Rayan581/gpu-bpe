"""
Computation layer: worker implementation for distributed training.

Each worker:
- Pulls parameters from parameter server
- Processes micro-batches
- Computes gradients
- Pushes gradients to parameter server
- Applies quantization and error feedback
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable
import time


class TrainingWorker:
    """
    Single worker in distributed training system.

    Implements:
    - parameter pulling from server
    - gradient computation and push
    - quantization via compression module
    - staleness tracking
    """

    def __init__(
        self,
        worker_id: int,
        model: nn.Module,
        parameter_server,
        tokenizer,
        device: torch.device = None,
        compression_fn: Optional[Callable] = None
    ):
        """
        Args:
            worker_id: unique worker identifier
            model: model to train (copy for this worker)
            parameter_server: reference to parameter server
            tokenizer: tokenizer for preprocessing
            device: compute device
            compression_fn: quantization function (optional)
        """
        self.worker_id = worker_id
        self.model = model.to(device or torch.device("cpu"))
        self.param_server = parameter_server
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.compression_fn = compression_fn

        # Training state
        self.train_steps = 0
        self.param_version = 0
        self.last_sync_step = 0
        self.staleness = 0

        # Optimizer (local)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

        # Statistics
        self.grad_norm_history = []
        self.loss_history = []
        self.update_times = []

    def pull_parameters(self):
        """Fetch latest parameters from parameter server."""
        start_time = time.time()

        params, version = self.param_server.pull_parameters(
            worker_id=self.worker_id
        )

        # Update model parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in params:
                    param.copy_(params[name])

        self.param_version = version
        self.staleness = self.train_steps - self.last_sync_step

        fetch_time = time.time() - start_time
        return fetch_time

    def compute_gradients(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        Compute gradients on a batch.

        Args:
            batch: (input_ids, labels)
            criterion: loss function

        Returns:
            loss value
        """
        input_ids, labels = batch

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(input_ids)

        # Loss
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward
        loss.backward()

        return loss.item()

    def push_gradients(self, apply_compression: bool = True) -> Dict:
        """
        Extract gradients and push to parameter server.

        Args:
            apply_compression: whether to quantize gradients

        Returns:
            {layer_name -> gradient_stat}
        """
        start_time = time.time()

        # Collect gradients
        gradients = {}
        grad_stats = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.data
                gradients[name] = grad.clone().detach()

                # Compute statistics
                grad_stats[name] = {
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item() if grad.numel() > 1 else 0.0
                }

        # Apply compression if available
        if apply_compression and self.compression_fn is not None:
            quantized, scales = self.compression_fn(gradients)
            dequantized = {}
            for name, q_grad in quantized.items():
                # Simple dequantization (assuming INT8 was used)
                dequantized[name] = q_grad.float() / 127.0 / scales.get(name, 1.0)

            # Track quantization error
            errors = {
                name: gradients[name] - dequantized.get(name, gradients[name])
                for name in gradients
            }
            gradients = dequantized
        else:
            errors = {}

        # Push to parameter server
        self.param_server.push_gradient(
            worker_id=self.worker_id,
            step=self.step,
            gradients=gradients,
            staleness=self.staleness,
            timestamp=time.time()
        )

        # Record quantization errors for feedback
        for name, error in errors.items():
            self.param_server.add_error_feedback(name, error)

        push_time = time.time() - start_time
        self.update_times.append(push_time)

        return grad_stats

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        pull_params: bool = True,
        push_grads: bool = True
    ) -> Dict:
        """
        Single training step: pull, compute, push.

        Returns:
            {loss, grad_stats, fetch_time, push_time}
        """
        result = {}

        # Pull latest parameters
        if pull_params:
            result['fetch_time'] = self.pull_parameters()
        else:
            result['fetch_time'] = 0

        # Compute gradients
        loss = self.compute_gradients(batch, criterion)
        result['loss'] = loss
        self.loss_history.append(loss)

        # Push gradients
        if push_grads:
            grad_stats = self.push_gradients()
            result['grad_stats'] = grad_stats
        else:
            result['grad_stats'] = {}

        self.train_steps += 1

        return result

    def get_worker_state(self) -> Dict:
        """Snapshot of worker state."""
        return {
            'worker_id': self.worker_id,
            'step': self.train_steps,
            'param_version': self.param_version,
            'staleness': self.staleness,
            'avg_loss': sum(self.loss_history[-10:]) / max(1, len(self.loss_history[-10:])),
            'avg_update_time': sum(self.update_times[-10:]) / max(1, len(self.update_times[-10:])) if self.update_times else 0,
            'loss_history_len': len(self.loss_history)
        }


class LocalTrainer:
    """
    Standalone local trainer (for single-GPU baseline).

    Used for train_baseline.py without distributed overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device = None,
        use_compression: bool = False
    ):
        self.model = model.to(device or torch.device("cpu"))
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.use_compression = use_compression

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.step = 0
        self.loss_history = []

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Single training step.

        Returns:
            (loss, grad_norm)
        """
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()

        # Compute gradient norm
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += (param.grad ** 2).sum().item()
        grad_norm = grad_norm ** 0.5

        self.optimizer.step()

        self.step += 1
        self.loss_history.append(loss.item())

        return loss.item(), grad_norm

    def get_state(self) -> Dict:
        """Get trainer state."""
        return {
            'step': self.step,
            'avg_loss': sum(self.loss_history[-100:]) / max(1, len(self.loss_history[-100:])),
            'loss_history_len': len(self.loss_history)
        }


def test_worker():
    """Test worker functionality."""
    from parameter_server import ParameterServer

    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

    server = ParameterServer(model)
    worker = TrainingWorker(
        worker_id=0,
        model=model,
        parameter_server=server,
        tokenizer=None
    )

    # Create dummy batch
    batch = (
        torch.randn(4, 10),  # input_ids
        torch.randint(0, 2, (4,))  # labels
    )

    criterion = nn.CrossEntropyLoss()

    # Single step
    result = worker.step(batch, criterion)
    print(f"Step result: loss={result['loss']:.4f}")
    print(f"Worker state: {worker.get_worker_state()}\n")


if __name__ == "__main__":
    test_worker()
