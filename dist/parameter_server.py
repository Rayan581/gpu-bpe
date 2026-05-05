"""
Parameter service layer: RPC-based parameter server with error feedback buffer.

Implements:
- Parameter push/pull operations
- Error feedback accumulation across gradient steps
- Staleness-aware gradient aggregation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import queue


@dataclass
class ParameterUpdate:
    """Single parameter update in parameter server."""
    step: int
    worker_id: int
    layer_name: str
    gradient_batch: torch.Tensor
    staleness: int
    timestamp: float


class ErrorFeedbackStore:
    """
    Maintains error residuals from gradient quantization.

    Accumulates quantization error and adds back to gradients
    in subsequent steps (error feedback mechanism).
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.residuals = {}  # layer_name -> accumulated error tensor

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.residuals[name] = torch.zeros_like(
                    param.data,
                    device=self.device
                )

    def add_residual(self, layer_name: str, error: torch.Tensor):
        """Add quantization error to residual buffer."""
        if layer_name in self.residuals:
            self.residuals[layer_name] = self.residuals[layer_name] + error

    def get_residual(self, layer_name: str) -> torch.Tensor:
        """Get accumulated residual for layer."""
        if layer_name in self.residuals:
            return self.residuals[layer_name].clone()
        return torch.tensor(0.0, device=self.device)

    def clear_residual(self, layer_name: str):
        """Clear residual after applying."""
        if layer_name in self.residuals:
            self.residuals[layer_name].zero_()

    def get_stats(self) -> Dict:
        """Get residual magnitude statistics."""
        stats = {}
        for name, residual in self.residuals.items():
            stats[name] = {
                'magnitude': residual.norm().item(),
                'mean': residual.mean().item(),
                'std': residual.std().item() if residual.numel() > 1 else 0.0
            }
        return stats


class ParameterServer:
    """
    RPC-based parameter server for distributed training.

    Maintains:
    - Current model parameters (parameters)
    - Parameter version for staleness tracking
    - Error feedback residuals (error_feedback)
    - Gradient aggregation buffers (gradient_queue)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        enable_error_feedback: bool = True
    ):
        self.device = device or torch.device("cpu")
        self.model = model

        # Parameter management
        self.parameters = {}
        self.param_version = 0
        self._cache_parameters()

        # Error feedback
        self.enable_error_feedback = enable_error_feedback
        self.error_feedback = ErrorFeedbackStore(model, device=self.device)

        # Gradient aggregation
        self.gradient_queue = queue.Queue()
        self.gradient_buffers = {}  # step -> {layer_name -> accumulated_grad}
        self.gradient_counts = {}   # step -> {layer_name -> worker_count}

        # Statistics
        self.update_counts = {}  # layer_name -> count
        self.lock = threading.Lock()

    def _cache_parameters(self):
        """Cache current model parameters."""
        self.parameters = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.parameters[name] = param.data.clone().detach()

    def pull_parameters(
        self,
        worker_id: int,
        version_hint: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Worker pulls current parameters from server.

        Args:
            worker_id: requesting worker id
            version_hint: expected version (for staleness check)

        Returns:
            (parameters_dict, current_version)
        """
        with self.lock:
            params_copy = {
                name: param.clone().detach()
                for name, param in self.parameters.items()
            }
            return params_copy, self.param_version

    def push_gradient(
        self,
        worker_id: int,
        step: int,
        gradients: Dict[str, torch.Tensor],
        staleness: int = 0,
        timestamp: float = None
    ):
        """
        Worker pushes gradients to server.

        Args:
            worker_id: id of submitting worker
            step: training step
            gradients: {layer_name -> gradient_tensor}
            staleness: gradient staleness (current_step - gradient_step)
            timestamp: wall-clock submission time
        """
        if timestamp is None:
            import time
            timestamp = time.time()

        # Add to gradient buffer for aggregation
        with self.lock:
            if step not in self.gradient_buffers:
                self.gradient_buffers[step] = {}
                self.gradient_counts[step] = {}

            for layer_name, grad in gradients.items():
                if layer_name not in self.gradient_buffers[step]:
                    self.gradient_buffers[step][layer_name] = torch.zeros_like(grad)
                    self.gradient_counts[step][layer_name] = 0

                # Accumulate with staleness scaling (optional)
                # scale = 1.0 / (1.0 + staleness)
                scale = 1.0

                self.gradient_buffers[step][layer_name] = (
                    self.gradient_buffers[step][layer_name]
                    + grad * scale
                )
                self.gradient_counts[step][layer_name] += 1

                # Track update count
                if layer_name not in self.update_counts:
                    self.update_counts[layer_name] = 0
                self.update_counts[layer_name] += 1

    def aggregate_gradients(
        self,
        step: int,
        num_workers: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients from all workers.

        Args:
            step: training step
            num_workers: expected number of workers

        Returns:
            {layer_name -> averaged_gradient}
        """
        with self.lock:
            if step not in self.gradient_buffers:
                return {}

            aggregated = {}
            for layer_name, grad_sum in self.gradient_buffers[step].items():
                count = self.gradient_counts[step].get(layer_name, 1)
                aggregated[layer_name] = grad_sum / max(count, 1)

                # Add error feedback if enabled
                if self.enable_error_feedback:
                    residual = self.error_feedback.get_residual(layer_name)
                    aggregated[layer_name] = aggregated[layer_name] + residual

            # Clear old buffers
            if step in self.gradient_buffers:
                del self.gradient_buffers[step]
            if step in self.gradient_counts:
                del self.gradient_counts[step]

            return aggregated

    def apply_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        learning_rate: float = 1e-4
    ):
        """
        Apply aggregated gradients to parameters.

        Args:
            gradients: {layer_name -> gradient}
            learning_rate: learning rate for gradient descent
        """
        with self.lock:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in gradients:
                    grad = gradients[name]
                    param.data = param.data - learning_rate * grad

            # Update cache and version
            self._cache_parameters()
            self.param_version += 1

    def add_error_feedback(
        self,
        layer_name: str,
        error: torch.Tensor
    ):
        """
        Register quantization error for error feedback.

        Args:
            layer_name: which layer
            error: quantization error tensor
        """
        self.error_feedback.add_residual(layer_name, error)

    def get_server_state(self) -> Dict:
        """Snapshot of server state."""
        with self.lock:
            return {
                'param_version': self.param_version,
                'gradient_buffers': len(self.gradient_buffers),
                'error_feedback_stats': self.error_feedback.get_stats(),
                'update_counts': self.update_counts.copy(),
                'pending_steps': list(self.gradient_buffers.keys())
            }


def test_parameter_server():
    """Test parameter server functionality."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

    server = ParameterServer(model)

    # Simulate worker pulling parameters
    params_v0, version = server.pull_parameters(worker_id=0)
    print(f"Worker pulled params, version {version}, got {len(params_v0)} layers\n")

    # Simulate two workers pushing gradients
    grads_w0 = {
        name: torch.randn_like(param) * 0.01
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    grads_w1 = {
        name: torch.randn_like(param) * 0.01
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    server.push_gradient(0, step=0, gradients=grads_w0, staleness=0)
    server.push_gradient(1, step=0, gradients=grads_w1, staleness=1)
    print("Workers submitted gradients\n")

    # Aggregate and apply
    aggregated = server.aggregate_gradients(step=0, num_workers=2)
    print(f"Aggregated {len(aggregated)} layers\n")

    server.apply_gradients(aggregated, learning_rate=1e-4)
    print(f"Applied gradients, new param version: {server.param_version}\n")

    # Check server state
    state = server.get_server_state()
    print(f"Server state: {state}\n")


if __name__ == "__main__":
    test_parameter_server()
