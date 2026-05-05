"""
GPU-aware byte-level BPE tokenizer using the Parallel BPE_a algorithm.

This module ties together two halves of the parallel pipeline from
``tokenizer/bpe.py``:

  Training (Parallel BPE_a)
    Phase 1+2 — speculative merge selection on the corpus byte stream
                using a max-heap with shadow-array invalidation.
    Phase 3   — dependency DAG between selected merges.
    Phase 4   — topological wave assignment (independent merges grouped
                into "waves" that may be applied in parallel).

  Inference (wave-based encoding)
    For each new text, we apply the learned merges wave-by-wave; each
    wave is a single linear pass that rewrites the current token list.
    Total passes = ``dag_depth`` (typically << ``num_merges``), giving
    a large speedup over the naive "scan-and-merge-once-per-rule" loop.

Key design choices
  - All token IDs are integers throughout (no str() conversions).
  - Bytes 0-255 occupy ids 0-255; merges create new ids 256, 257, ...
  - During training we concatenate the corpus into a single byte stream
    (with a newline separator). This matches the algorithm in
    ``tokenizer/bpe.py`` exactly and makes the wave structure portable
    to inference.

API (unchanged from the previous version)
  - ``GPUBPETokenizer(vocab_size, use_gpu).bpe.train(texts, vocab_size, num_merges)``
  - ``tokenizer.encode(text) -> List[int]``
  - ``tokenizer.decode(token_ids) -> str``
  - ``tokenizer.bpe.bpe_merges``  (kept for compatibility — pair -> rank)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from tokenizer.bpe import (
    _speculative_select,
    _build_dag,
    _topo_levels,
    _apply_merge_wave,
)


# Module-level worker for ProcessPoolExecutor (must be picklable).
def _encode_chunk_worker(args):
    """Encode a list of texts using a given (merge_order, dag_levels)."""
    chunk, merge_order, dag_levels = args
    out = []
    for text in chunk:
        tokens = list(text.encode('utf-8'))
        if len(tokens) >= 2 and dag_levels:
            for level_indices in dag_levels:
                wave = [merge_order[i] for i in level_indices]
                tokens = _apply_merge_wave((tokens, wave))
        out.append(tokens)
    return out


class ByteLevelBPE:
    """
    Byte-level BPE tokenizer using Parallel BPE_a.

    Training selects merges with a speculative max-heap pass over the
    corpus byte stream and groups them into a DAG of independent waves.
    Inference applies the learned merges wave-by-wave, so each text is
    tokenized in ``dag_depth`` linear passes rather than ``num_merges``.
    """

    # Byte separator inserted between corpus documents during training.
    # A newline (0x0A) is a natural choice and means cross-document
    # merges look the same as in-document merges containing a newline.
    _DOC_SEP = 0x0A

    def __init__(self, vocab_size: int = 50257, gpu: bool = True):
        self.vocab_size = vocab_size
        self.use_gpu = gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # ── Backwards-compatible attributes ─────────────────────────────
        # (left_int, right_int) -> rank  — the notebook uses len(...) on this
        self.bpe_merges: Dict[Tuple[int, int], int] = {}

        # ── Parallel BPE_a state ────────────────────────────────────────
        # merge_order[i] = ((a, b), new_id) — speculatively-selected order
        self.merge_order: List[Tuple[Tuple[int, int], int]] = []
        # dag_levels[k] = list of merge indices forming the k-th wave
        self.dag_levels: List[List[int]] = []

        # Token-id <-> pair lookups
        self.merge_to_id: Dict[Tuple[int, int], int] = {}
        self.id_to_merge: Dict[int, Tuple[int, int]] = {}

        # Vocab size after training (256 base + len(merge_order)).
        self.vocab_size_actual = 256

        # Special tokens
        self.special_tokens = {50256: '<|endoftext|>'}

        # Compatibility shim — some callers expect a `vocab` dict.
        # Real ids are integers, so this is just a self-map for safety.
        self.byte_decoder = {i: bytes([i]) for i in range(256)}
        self.byte_encoder = {bytes([i]): i for i in range(256)}

    @property
    def vocab(self):
        # The model uses ``vocab_size`` for its embedding matrix shape;
        # what we expose here just needs to be dict-like.
        return {tid: tid for tid in range(max(self.vocab_size_actual, 50257))}

    # ───────────────────────────────────────────────────────────────────
    # Training — Parallel BPE_a
    # ───────────────────────────────────────────────────────────────────

    def train(
        self,
        texts: List[str],
        vocab_size: int = 50257,
        num_merges: int = 50000,
        max_workers: int = 4,
    ) -> Tuple[int, int]:
        """
        Train the BPE merge table using Parallel BPE_a.

        Args:
            texts: list of training text strings.
            vocab_size: target vocabulary size (caps the merge count).
            num_merges: number of BPE merges to attempt to learn.
            max_workers: kept for API compatibility (no parallel
                workers are used during training itself; multiprocess
                speedup is exposed at inference via ``encode_parallel``).

        Returns:
            (dag_depth, num_waves) — equal here, retained for the
            existing call signature in the notebook.
        """
        self.vocab_size = vocab_size
        # Reserve space for byte vocab (256) + endoftext (1).
        num_merges = min(num_merges, vocab_size - 256 - 1)

        # ── Phase 0: corpus byte stream ───────────────────────────────
        print(f"  [0/4] Building byte stream from {len(texts)} texts...")
        flat_tokens: List[int] = []
        for text in texts:
            flat_tokens.extend(text.encode('utf-8'))
            flat_tokens.append(self._DOC_SEP)
        stream_len = len(flat_tokens)
        print(f"        Stream length: {stream_len:,} bytes")

        if stream_len < 2:
            print("  Stream too short — skipping training.")
            return 0, 0

        # ── Phase 1+2: speculative merge selection ───────────────────
        print(f"  [1/4] Speculative merge selection (target {num_merges} merges)...")
        base_next_id = 256
        merge_order, _ = _speculative_select(
            flat_tokens, num_merges, base_next_id
        )
        print(f"        {len(merge_order)} merges selected")

        # ── Phase 3: dependency DAG ──────────────────────────────────
        print("  [2/4] Building dependency DAG...")
        deps = _build_dag(merge_order)

        # ── Phase 4: topological wave assignment ─────────────────────
        print("  [3/4] Computing topological waves...")
        levels = _topo_levels(deps) if merge_order else []
        dag_depth = len(levels)
        if merge_order:
            avg = len(merge_order) / max(dag_depth, 1)
            parallelism = (1 - dag_depth / len(merge_order)) * 100
            print(
                f"        DAG depth: {dag_depth} waves "
                f"(avg {avg:.1f} merges/wave, {parallelism:.1f}% parallelism)"
            )

        # ── Persist ──────────────────────────────────────────────────
        self.merge_order = merge_order
        self.dag_levels = levels
        self.bpe_merges = {pair: rank for rank, (pair, _) in enumerate(merge_order)}
        self.merge_to_id = {pair: new_id for pair, new_id in merge_order}
        self.id_to_merge = {new_id: pair for pair, new_id in merge_order}
        self.vocab_size_actual = base_next_id + len(merge_order)

        print(
            f"  [4/4] Done. {len(merge_order)} merges, "
            f"{dag_depth} waves, vocab = {self.vocab_size_actual}"
        )
        return dag_depth, dag_depth

    # ───────────────────────────────────────────────────────────────────
    # Inference — wave-based encoding
    # ───────────────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> List[int]:
        """Encode one text using ``dag_depth`` linear passes (one per wave)."""
        if not text:
            return []
        tokens: List[int] = list(text.encode('utf-8'))
        if len(tokens) < 2 or not self.dag_levels:
            return tokens

        for level_indices in self.dag_levels:
            wave = [self.merge_order[i] for i in level_indices]
            tokens = _apply_merge_wave((tokens, wave))
        return tokens

    def encode(
        self,
        texts: Union[str, List[str]],
        return_tensors: bool = False,
    ) -> Union[List[int], Tuple[List[List[int]], List[Tuple[int, int]]]]:
        """
        Encode text(s) to token IDs.

        - ``str`` input  -> ``List[int]``
        - ``List[str]``  -> ``(List[List[int]], offsets)`` or
                            ``(Tensor, offsets)`` if ``return_tensors=True``
        """
        if isinstance(texts, str):
            return self._encode_text(texts)

        all_ids = [self._encode_text(t) for t in texts]
        offsets = [(i, len(ids)) for i, ids in enumerate(all_ids)]

        if return_tensors:
            max_len = max((len(ids) for ids in all_ids), default=0)
            padded = [ids + [50256] * (max_len - len(ids)) for ids in all_ids]
            return torch.tensor(padded, device=self.device), offsets
        return all_ids, offsets

    def encode_batch_parallel(
        self,
        texts: List[str],
        max_workers: int = 4,
    ) -> List[List[int]]:
        """
        Multi-process batch encoding.

        Each worker encodes a chunk of texts independently. Within a
        single text we still exploit the wave structure, so this gives
        two-level parallelism: across texts (processes) and within
        a text (waves).
        """
        if not texts:
            return []
        if max_workers <= 1 or len(texts) < max_workers:
            return [self._encode_text(t) for t in texts]

        chunk_size = max(1, (len(texts) + max_workers - 1) // max_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        merge_order = self.merge_order
        dag_levels = self.dag_levels
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(
                _encode_chunk_worker,
                [(chunk, merge_order, dag_levels) for chunk in chunks],
            ))
        return [ids for chunk_result in results for ids in chunk_result]

    # ───────────────────────────────────────────────────────────────────
    # Decoding — recursive expansion of merged tokens to bytes
    # ───────────────────────────────────────────────────────────────────

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        out = bytearray()
        for tid in token_ids:
            if tid == 50256:  # endoftext
                break
            self._expand_token(int(tid), out)

        try:
            return out.decode('utf-8')
        except UnicodeDecodeError:
            return out.decode('utf-8', errors='replace')

    def _expand_token(self, token_id: int, out: bytearray):
        """Recursively expand a (possibly merged) token to its byte sequence."""
        if token_id < 256:
            out.append(token_id)
        elif token_id in self.id_to_merge:
            a, b = self.id_to_merge[token_id]
            self._expand_token(a, out)
            self._expand_token(b, out)
        # else: unknown id — silently skip

    def get_vocab_size(self) -> int:
        return max(self.vocab_size_actual, 50257)


# ────────────────────────────────────────────────────────────────────────
# GPUBPETokenizer — wrapper with batch processing and GPU-staged tensors
# ────────────────────────────────────────────────────────────────────────

class GPUBPETokenizer:
    """
    Wrapper around ``ByteLevelBPE`` exposing batch encoding that stages
    tokens on the GPU for downstream training/inference without an extra
    host-to-device copy per batch.
    """

    def __init__(self, vocab_size: int = 50257, use_gpu: bool = True):
        self.bpe = ByteLevelBPE(vocab_size=vocab_size, gpu=use_gpu)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        pad_token_id: int = 50256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of texts with padding/truncation to ``max_length``.

        Returns:
            (token_ids: [B, T], attention_mask: [B, T]) on ``self.device``.
        """
        token_lists, _ = self.bpe.encode(texts)
        if max_length is None:
            max_length = max((len(ids) for ids in token_lists), default=0)

        padded, masks = [], []
        for ids in token_lists:
            if len(ids) > max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids) + [0] * (max_length - len(ids))
            ids = ids + [pad_token_id] * (max_length - len(ids))
            padded.append(ids)
            masks.append(mask)

        token_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        return token_ids, attention_mask

    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy().tolist()
        return [self.bpe.decode(ids) for ids in token_ids]

    def encode(self, text):
        return self.bpe.encode(text)

    def decode(self, token_ids):
        return self.bpe.decode(token_ids)

    def encode_parallel(
        self,
        texts: List[str],
        max_workers: int = 4,
    ) -> List[List[int]]:
        """Multi-process batch encoding (uses the parallel BPE_a wave structure)."""
        return self.bpe.encode_batch_parallel(texts, max_workers=max_workers)

    def train(
        self,
        texts: List[str],
        max_workers: int = 4,
    ) -> Tuple[int, int]:
        """Train tokenizer on corpus. Returns (dag_depth, num_waves)."""
        return self.bpe.train(
            texts,
            vocab_size=self.bpe.vocab_size,
            max_workers=max_workers,
        )
