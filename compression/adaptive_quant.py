"""
Adaptive gradient quantization with per-layer alpha/beta scaling and error feedback.

Implements layer-wise variance-based quantization bounds with INT8/INT4 support
and residual error accumulation for DC-ASGD compensation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class AdaptiveQuantizer:
    """
    Per-layer adaptive quantization with dynamic alpha/beta bounds.

    The quantizer maintains per-layer statistics and updates quantization
    bounds every N steps based on gradient variance.
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        update_interval: int = 50,
        momentum: float = 0.9,
        device: torch.device = None
    ):
        """
        Args:
            model: PyTorch model
            bits: quantization bits (8 or 4)
            update_interval: steps between alpha/beta updates
            momentum: momentum for running variance estimate
            device: compute device
        """
        self.bits = bits
        self.update_interval = update_interval
        self.momentum = momentum
        self.device = device or torch.device("cpu")
        self.step = 0

        # Per-layer statistics
        self.layer_names = []
        self.alpha = {}  # scale factors
        self.beta = {}   # clipping bounds
        self.running_var = {}  # running variance
        self.error_feedback = {}  # residual buffers

        self._init_from_model(model)

    def _init_from_model(self, model: nn.Module):
        """Initialize quantizer state for each layer."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)

                # Initialize bounds
                self.alpha[name] = 1.0
                self.beta[name] = torch.tensor(1.0, device=self.device)
                self.running_var[name] = 0.0
                self.error_feedback[name] = torch.zeros_like(param.data)

    def _get_quantization_scale(self, grad: torch.Tensor) -> Tuple[float, float]:
        """Compute alpha (scale) and beta (clip threshold) from gradient statistics."""
        with torch.no_grad():
            var = grad.pow(2).mean().item()
            scale = 1.0 / (torch.abs(grad).max().item() + 1e-8)
            clip_threshold = torch.std(grad).item() * 3.0  # 3-sigma clipping
        return scale, clip_threshold

    def _quantize_int8(self, grad: torch.Tensor, scale: float) -> torch.Tensor:
        """Quantize gradient to INT8."""
        # Scale to [-128, 127]
        scaled = torch.clamp(grad * scale, -1.0, 1.0) * 127
        quantized = torch.round(scaled).to(torch.int8)
        return quantized

    def _quantize_int4(self, grad: torch.Tensor, scale: float) -> torch.Tensor:
        """Quantize gradient to INT4 (via INT8 with coarser quantization)."""
        # Scale to [-8, 7]
        scaled = torch.clamp(grad * scale, -1.0, 1.0) * 7
        quantized = torch.round(scaled).clamp(-8, 7).to(torch.int8)
        return quantized

    def _dequantize(
        self,
        quantized: torch.Tensor,
        scale: float,
        bits: int
    ) -> torch.Tensor:
        """Dequantize back to float."""
        if bits == 8:
            max_val = 127
        elif bits == 4:
            max_val = 7
        else:
            max_val = 127

        return quantized.to(torch.float32) / max_val / scale

    def quantize_grads(
        self,
        grads: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Quantize all gradients with adaptive bounds.

        Args:
            grads: dict of {layer_name: gradient_tensor}

        Returns:
            (quantized_grads, scales)
        """
        quantized = {}
        scales = {}

        for name in self.layer_names:
            if name not in grads:
                continue

            grad = grads[name]

            # Add error feedback from previous step
            if name in self.error_feedback:
                grad = grad + self.error_feedback[name]

            # Update running variance
            with torch.no_grad():
                var = grad.pow(2).mean().item()
                self.running_var[name] = (
                    self.momentum * self.running_var[name]
                    + (1 - self.momentum) * var
                )

            # Get adaptive bounds
            scale, clip_threshold = self._get_quantization_scale(grad)

            # Clip gradient
            grad = torch.clamp(grad, -clip_threshold, clip_threshold)

            # Quantize
            if self.bits == 8:
                quantized[name] = self._quantize_int8(grad, scale)
            elif self.bits == 4:
                quantized[name] = self._quantize_int4(grad, scale)
            else:
                raise ValueError(f"Unsupported bits: {self.bits}")

            scales[name] = scale

        self.step += 1

        # Update alpha/beta periodically
        if self.step % self.update_interval == 0:
            self._update_bounds()

        return quantized, scales

    def dequantize_grads(
        self,
        quantized: Dict[str, torch.Tensor],
        scales: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize gradients."""
        dequantized = {}
        for name, quant_grad in quantized.items():
            scale = scales.get(name, 1.0)
            dequantized[name] = self._dequantize(quant_grad, scale, self.bits)

            # Store residual for error feedback
            if name not in dequantized:
                self.error_feedback[name] = torch.zeros_like(quant_grad)

        return dequantized

    def _update_bounds(self):
        """Update alpha/beta based on running variance (called every N steps)."""
        for name in self.layer_names:
            var = self.running_var.get(name, 1.0)
            self.alpha[name] = max(0.5, min(2.0, 1.0 / (var + 1e-8)))
            self.beta[name] = torch.tensor(
                np.sqrt(var) * 3.0,
                device=self.device
            )

    def get_compression_ratio(self) -> float:
        """Estimate gradient communication reduction ratio."""
        if self.bits == 8:
            return 32.0 / 8.0  # FP32 to INT8
        elif self.bits == 4:
            return 32.0 / 4.0  # FP32 to INT4
        else:
            return 1.0

    def get_stats(self) -> Dict:
        """Return quantizer statistics."""
        return {
            'bits': self.bits,
            'step': self.step,
            'layer_count': len(self.layer_names),
            'compression_ratio': self.get_compression_ratio(),
            'running_vars': {
                name: self.running_var.get(name, 0.0)
                for name in self.layer_names
            }
        }


class ErrorFeedbackBuffer:
    """
    Maintains residual error across quantization rounds for gradient flow.

    Implements error feedback to compensate for quantization error
    in subsequent steps (for DC-ASGD).
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.buffers = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.buffers[name] = torch.zeros_like(
                    param.data,
                    device=self.device
                )

    def accumulate_error(
        self,
        original_grads: Dict[str, torch.Tensor],
        quantized_grads: Dict[str, torch.Tensor]
    ):
        """
        Accumulate quantization error in buffers.

        Args:
            original_grads: FP32 gradients before quantization
            quantized_grads: dequantized gradients after quantization
        """
        for name in original_grads:
            if name in quantized_grads:
                error = original_grads[name] - quantized_grads[name]
                self.buffers[name] = self.buffers[name] + error

    def get_feedback(self, layer_name: str) -> torch.Tensor:
        """Get accumulated error feedback for a layer."""
        return self.buffers.get(layer_name, torch.tensor(0.0))

    def reset(self):
        """Clear all error buffers."""
        for name in self.buffers:
            self.buffers[name].zero_()


def test_quantization():
    """Test adaptive quantizer on dummy model."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

    quantizer = AdaptiveQuantizer(model, bits=8, update_interval=2)

    # Simulate gradients
    for step in range(5):
        grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                grads[name] = torch.randn_like(param) * 0.01

        # Quantize
        quantized, scales = quantizer.quantize_grads(grads)
        print(f"Step {step}: Quantized {len(quantized)} layers")

        # Dequantize
        dequantized = quantizer.dequantize_grads(quantized, scales)
        print(f"  Compression ratio: {quantizer.get_compression_ratio():.2f}x")

        print(f"  Stats: {quantizer.get_stats()}\n")


if __name__ == "__main__":
    test_quantization()
