"""Tokenizer module: GPU-aware BPE with HSG digit-span locking."""

from .gpu_bpe import GPUBPETokenizer, ByteLevelBPE
from .hsg import SemanticGuardedTokenizer, DigitSpanLocker
from .sequential_bpe import SequentialBPETokenizer
from .cuda_bpe import (
    CudaBPETokenizer,
    cuda_bpe,
    standard_bpe,
    gpu_count_pairs,
    gpu_apply_wave,
    CUDA_AVAILABLE,
)

__all__ = [
    'GPUBPETokenizer',
    'ByteLevelBPE',
    'SemanticGuardedTokenizer',
    'DigitSpanLocker',
    'SequentialBPETokenizer',
    'CudaBPETokenizer',
    'cuda_bpe',
    'standard_bpe',
    'gpu_count_pairs',
    'gpu_apply_wave',
    'CUDA_AVAILABLE',
]
