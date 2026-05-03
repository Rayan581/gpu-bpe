"""Compression module: adaptive quantization with error feedback."""

from .adaptive_quant import AdaptiveQuantizer, ErrorFeedbackBuffer

__all__ = [
    'AdaptiveQuantizer',
    'ErrorFeedbackBuffer'
]
