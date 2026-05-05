"""
CuPy-based CUDA BPE Tokenizer (memory-efficient, NVRTC-free)
=============================================================
Compatible with Google Colab / Tesla T4 / any CuPy install.
No RawModule / RawKernel — zero NVRTC dependency.

Key design
----------
- gpu_count_pairs   : pair frequency counting via cp.unique on GPU
- gpu_apply_wave    : wave application via cp.searchsorted (O(n log W),
                      O(n + W) memory — no broadcast match matrix)
- _speculative_select_gpu : Phase 1+2 of Parallel BPE_a; initial pair
                      counts on GPU, shadow-array invalidation on CPU
- _build_dag / _topo_levels : Phase 3+4; dependency DAG + wave assignment
- cuda_bpe          : full pipeline — train + encode on a text string
- standard_bpe      : sequential baseline (CPU, character-level)
- CudaBPETokenizer  : class wrapper for byte-level corpus training,
                      compatible with the rest of the project pipeline

Falls back to NumPy transparently when CuPy is unavailable so the
module can be imported on CPU-only machines without errors.
"""

from __future__ import annotations

import time
import collections
import heapq
from typing import Dict, List, Tuple, Optional

try:
    import cupy as cp
    import numpy as np
    CUDA_AVAILABLE = True
except ImportError:
    import numpy as np
    cp = np          # shim: NumPy as fallback
    CUDA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# GPU primitives
# ─────────────────────────────────────────────────────────────────────────────

def gpu_count_pairs(
    d_tokens: "cp.ndarray",
    vocab_cap: int,
) -> Dict[Tuple[int, int], int]:
    """Count adjacent pair frequencies entirely on the GPU via cp.unique."""
    n = len(d_tokens)
    if n < 2:
        return {}
    lefts  = d_tokens[:-1].astype(cp.int64)
    rights = d_tokens[1:].astype(cp.int64)
    keys   = lefts * vocab_cap + rights
    unique_keys, counts = cp.unique(keys, return_counts=True)
    uk = unique_keys.get() if CUDA_AVAILABLE else unique_keys
    uc = counts.get()      if CUDA_AVAILABLE else counts
    result: Dict[Tuple[int, int], int] = {}
    for k, c in zip(uk, uc):
        result[(int(k) // vocab_cap, int(k) % vocab_cap)] = int(c)
    return result


def gpu_apply_wave(
    d_tokens: "cp.ndarray",
    wave: List[Tuple[Tuple[int, int], int]],
    vocab_cap: int,
) -> "cp.ndarray":
    """
    Apply one wave of independent BPE merges on the GPU.

    Uses cp.searchsorted for O(n log W) lookup instead of a broadcast
    match matrix — safe for large sequences and large waves.
    """
    n = len(d_tokens)
    if n < 2 or not wave:
        return d_tokens

    lefts     = d_tokens[:-1].astype(cp.int64)
    rights    = d_tokens[1:].astype(cp.int64)
    pair_keys = lefts * vocab_cap + rights

    wave_keys_np = np.array(
        [p[0] * vocab_cap + p[1] for (p, _) in wave], dtype=np.int64
    )
    wave_vals_np = np.array([nid for (_, nid) in wave], dtype=np.int32)

    sort_order   = np.argsort(wave_keys_np)
    wave_keys_np = wave_keys_np[sort_order]
    wave_vals_np = wave_vals_np[sort_order]

    d_wave_keys = cp.asarray(wave_keys_np)
    d_wave_vals = cp.asarray(wave_vals_np)

    idx      = cp.searchsorted(d_wave_keys, pair_keys, side='left')
    in_range = idx < len(d_wave_keys)
    idx_safe = cp.clip(idx, 0, len(d_wave_keys) - 1)
    exact    = d_wave_keys[idx_safe] == pair_keys
    hit_mask = in_range & exact

    out_tokens = d_tokens.copy().astype(cp.int32)
    alive      = cp.ones(n, dtype=cp.bool_)

    hit_positions = cp.where(hit_mask)[0]
    if hit_positions.size > 0:
        new_ids_at_hits = d_wave_vals[idx_safe[hit_positions]]
        out_tokens[hit_positions] = new_ids_at_hits
        right_positions = hit_positions + 1
        right_positions = right_positions[right_positions < n]
        alive[right_positions] = False

    alive_int   = alive.view(cp.int8)
    scan        = cp.cumsum(alive_int).astype(cp.int32) - 1
    new_len     = int(alive_int.sum())
    out_compact = cp.empty(new_len, dtype=cp.int32)
    out_compact[scan[alive]] = out_tokens[alive]
    return out_compact


# ─────────────────────────────────────────────────────────────────────────────
# Parallel BPE_a — phases 1-4
# ─────────────────────────────────────────────────────────────────────────────

def _speculative_select_gpu(
    tokens_orig: List[int],
    num_merges: int,
    base_next_id: int,
    vocab_cap: int,
) -> Tuple[list, dict]:
    """
    Phase 1+2: speculative merge selection.
    Initial pair counts via GPU; shadow-array invalidation on CPU.
    """
    d_tok          = cp.asarray(np.array(tokens_orig, dtype=np.int32))
    pair_freq_dict = gpu_count_pairs(d_tok, vocab_cap)

    shadow    = list(tokens_orig)
    dead: set = set()
    pair_locs: Dict = collections.defaultdict(list)
    for i in range(len(tokens_orig) - 1):
        pair_locs[(tokens_orig[i], tokens_orig[i + 1])].append(i)

    heap: list = []
    counter = 0
    for pair, freq in pair_freq_dict.items():
        heapq.heappush(heap, (-freq, counter, pair))
        counter += 1

    invalidated: set  = set()
    next_id           = base_next_id
    merge_order: list = []
    new_vocab:  dict  = {}

    try:
        from tokenizer._progress import get_tqdm as _get_tqdm
        _tqdm = _get_tqdm()
    except ImportError:
        _tqdm = None

    t_start = time.perf_counter()
    pbar = (
        _tqdm(total=num_merges, desc="GPU BPE (speculative)", unit="merge",
              dynamic_ncols=True)
        if _tqdm is not None else None
    )

    for merge_idx in range(num_merges):
        t0 = time.perf_counter()

        best_pair = None
        heap_pops = 0
        while heap:
            _, _, p = heapq.heappop(heap)
            heap_pops += 1
            if p in invalidated:
                continue
            act = sum(
                1 for pos in pair_locs[p]
                if pos     not in dead
                and pos + 1 not in dead
                and shadow[pos]     == p[0]
                and shadow[pos + 1] == p[1]
            )
            if act >= 2:
                best_pair = p
                break
        if not best_pair:
            if pbar is not None:
                pbar.close()
            print(f"  No more valid pairs after {merge_idx} merges "
                  f"({time.perf_counter()-t_start:.2f}s)")
            break

        nid = next_id
        next_id += 1
        new_vocab[nid] = best_pair
        merge_order.append((best_pair, nid))
        invalidated.add(best_pair)

        valid_positions = [
            pos for pos in pair_locs[best_pair]
            if pos     not in dead
            and pos + 1 not in dead
            and shadow[pos]     == best_pair[0]
            and shadow[pos + 1] == best_pair[1]
        ]
        for pos in valid_positions:
            lp = pos - 1
            rp = pos + 2
            while lp >= 0 and lp in dead:
                lp -= 1
            while rp < len(shadow) and rp in dead:
                rp += 1

            if lp >= 0:
                invalidated.add((shadow[lp], shadow[pos]))
            if rp < len(shadow):
                invalidated.add((shadow[pos + 1], shadow[rp]))

            shadow[pos] = nid
            dead.add(pos + 1)

            if lp >= 0:
                pair_locs[(shadow[lp], nid)].append(lp)
            if rp < len(shadow):
                pair_locs[(nid, shadow[rp])].append(pos)

        step_ms = (time.perf_counter() - t0) * 1000
        if pbar is not None:
            pbar.set_postfix({
                "merge":    merge_idx + 1,
                "heap_pop": heap_pops,
                "valid":    len(valid_positions),
                "dead":     len(dead),
                "ms":       f"{step_ms:.1f}",
            })
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    elapsed = time.perf_counter() - t_start
    n = len(merge_order)
    print(f"  _speculative_select done: {n} merges in {elapsed:.2f}s "
          f"({elapsed/max(n,1)*1000:.1f} ms/merge avg, "
          f"dead positions: {len(dead):,})")

    return merge_order, new_vocab


def _build_dag(merge_order: list) -> dict:
    """Phase 3: build dependency DAG between selected merges."""
    prod: dict = {}
    deps: dict = {i: set() for i in range(len(merge_order))}
    for idx, (pair, nt) in enumerate(merge_order):
        for t in pair:
            if t in prod:
                deps[idx].add(prod[t])
        prod[nt] = idx
    return deps


def _topo_levels(deps: dict) -> list:
    """Phase 4: topological wave assignment — independent merges per level."""
    n        = len(deps)
    level    = [-1] * n
    children: dict = collections.defaultdict(list)
    in_deg   = [0] * n

    for node, parents in deps.items():
        for p in parents:
            children[p].append(node)
            in_deg[node] += 1

    q = collections.deque(node for node in range(n) if in_deg[node] == 0)
    for node in q:
        level[node] = 0

    while q:
        u = q.popleft()
        for v in children[u]:
            in_deg[v] -= 1
            level[v]   = max(level[v], level[u] + 1)
            if in_deg[v] == 0:
                q.append(v)

    if not level:
        return []

    max_lv = max(level)
    levels = [[] for _ in range(max_lv + 1)]
    for node, lv in enumerate(level):
        levels[lv].append(node)
    return levels


# ─────────────────────────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────────────────────────

def cuda_bpe(
    text: str,
    num_merges: int,
    vocab_cap: Optional[int] = None,
) -> Tuple[dict, list, list, int]:
    """
    Full Parallel BPE_a pipeline on a single text string (character-level).

    Returns
    -------
    vocab       : id -> symbol string
    merge_order : list of ((left_id, right_id), new_id)
    tokens      : encoded token list
    dag_depth   : number of waves (= DAG depth)
    """
    c2i: dict = {}
    i2s: dict = {}
    tokens: list = []
    for c in text:
        if c not in c2i:
            tid      = len(c2i)
            c2i[c]   = tid
            i2s[tid] = c
        tokens.append(c2i[c])

    base_id = len(c2i)
    vocab   = dict(i2s)
    v_cap   = vocab_cap or (base_id + num_merges + 10)

    merge_order, new_vocab = _speculative_select_gpu(
        tokens, num_merges, base_id, v_cap
    )

    def resolve(t: int) -> str:
        if t in vocab:
            return vocab[t]
        l, r = new_vocab[t]
        return resolve(l) + resolve(r)

    for nid in sorted(new_vocab):
        vocab[nid] = resolve(nid)

    levels  = _topo_levels(_build_dag(merge_order))
    d_tokens = cp.asarray(np.array(tokens, dtype=np.int32))
    for lv in levels:
        wave     = [merge_order[i] for i in lv]
        d_tokens = gpu_apply_wave(d_tokens, wave, v_cap)

    final = d_tokens.get().tolist() if CUDA_AVAILABLE else d_tokens.tolist()
    return vocab, merge_order, final, len(levels)


def standard_bpe(
    text: str,
    num_merges: int,
) -> Tuple[dict, list, list]:
    """
    Sequential BPE baseline (CPU, character-level).
    Used as the comparison target in benchmarks.

    Returns
    -------
    vocab, merge_order, tokens
    """
    c2i: dict = {}
    i2s: dict = {}
    tokens: list = []
    for c in text:
        if c not in c2i:
            tid      = len(c2i)
            c2i[c]   = tid
            i2s[tid] = c
        tokens.append(c2i[c])

    next_id = len(c2i)
    merges: list = []
    vocab   = dict(i2s)

    try:
        from tokenizer._progress import get_tqdm as _get_tqdm
        _tqdm = _get_tqdm()
    except ImportError:
        _tqdm = None

    t_start = time.perf_counter()
    pbar = (
        _tqdm(total=num_merges, desc="standard_bpe", unit="merge",
              dynamic_ncols=True)
        if _tqdm is not None else None
    )

    for merge_idx in range(num_merges):
        t0 = time.perf_counter()
        f: dict = collections.defaultdict(int)
        for i in range(len(tokens) - 1):
            f[(tokens[i], tokens[i + 1])] += 1
        if not f:
            break
        best = max(f, key=f.get)
        if f[best] < 2:
            break

        nid      = next_id
        next_id += 1
        vocab[nid] = vocab[best[0]] + vocab[best[1]]
        merges.append((best, nid))

        new_t: list = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i]     == best[0]
                and tokens[i + 1] == best[1]
            ):
                new_t.append(nid)
                i += 2
            else:
                new_t.append(tokens[i])
                i += 1
        tokens = new_t

        step_ms = (time.perf_counter() - t0) * 1000
        if pbar is not None:
            pbar.set_postfix({
                "merge": merge_idx + 1,
                "freq":  f[best],
                "toks":  len(tokens),
                "ms/merge": f"{step_ms:.1f}",
            })
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    elapsed = time.perf_counter() - t_start
    print(f"  standard_bpe done: {len(merges)} merges in {elapsed:.2f}s "
          f"({elapsed/max(len(merges),1)*1000:.1f} ms/merge avg)")

    return vocab, merges, tokens


# ─────────────────────────────────────────────────────────────────────────────
# CudaBPETokenizer — byte-level class wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CudaBPETokenizer:
    """
    Byte-level BPE tokenizer backed by the CuPy CUDA pipeline.

    Training : concatenates corpus into a single byte stream, runs
               _speculative_select_gpu + DAG + wave assignment.
    Encoding : applies learned waves via gpu_apply_wave (GPU) or
               a pure-Python fallback on CPU.
    Decoding : recursive byte expansion.

    This is a drop-in replacement for GPUBPETokenizer in the notebook
    with true GPU-accelerated pair counting and wave application.
    """

    _DOC_SEP = 0x0A   # newline separator between documents

    def __init__(self, vocab_size: int = 50257):
        self.vocab_size      = vocab_size
        self.merge_order:  list = []
        self.dag_levels:   list = []
        self.id_to_merge:  dict = {}
        self.vocab_cap:    int  = vocab_size + 10
        self._trained      = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        texts: List[str],
        num_merges: int = 2000,
    ) -> Tuple[int, int]:
        """
        Train on a list of text strings.

        Returns (dag_depth, num_waves) — equal here.
        """
        num_merges = min(num_merges, self.vocab_size - 256 - 1)

        print(f"  [0/4] Building byte stream from {len(texts)} texts...")
        t0_phase = time.perf_counter()
        flat: List[int] = []
        for text in texts:
            flat.extend(text.encode('utf-8'))
            flat.append(self._DOC_SEP)
        print(f"        Stream length: {len(flat):,} bytes  "
              f"({time.perf_counter()-t0_phase:.2f}s)")

        if len(flat) < 2:
            return 0, 0

        self.vocab_cap = 256 + num_merges + 10

        print(f"  [1/4] Speculative merge selection (target {num_merges} merges)...")
        t0_phase = time.perf_counter()
        self.merge_order, self.id_to_merge = _speculative_select_gpu(
            flat, num_merges, 256, self.vocab_cap
        )
        print(f"        {len(self.merge_order)} merges selected  "
              f"({time.perf_counter()-t0_phase:.2f}s)")

        print("  [2/4] Building dependency DAG...")
        t0_phase = time.perf_counter()
        deps = _build_dag(self.merge_order)
        print(f"        DAG built  ({time.perf_counter()-t0_phase:.2f}s)")

        print("  [3/4] Computing topological waves...")
        t0_phase = time.perf_counter()
        self.dag_levels = _topo_levels(deps) if self.merge_order else []
        dag_depth = len(self.dag_levels)
        print(f"        Levels computed  ({time.perf_counter()-t0_phase:.2f}s)")

        if self.merge_order:
            avg         = len(self.merge_order) / max(dag_depth, 1)
            parallelism = (1 - dag_depth / len(self.merge_order)) * 100
            print(
                f"        DAG depth: {dag_depth} waves "
                f"(avg {avg:.1f} merges/wave, {parallelism:.1f}% parallelism)"
            )

        self._trained = True
        print(f"  [4/4] Done. {len(self.merge_order)} merges, {dag_depth} waves")
        return dag_depth, dag_depth

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Encode a single string to token IDs."""
        if not text:
            return []
        tokens = list(text.encode('utf-8'))
        if len(tokens) < 2 or not self.dag_levels:
            return tokens

        d_tokens = cp.asarray(np.array(tokens, dtype=np.int32))
        for level_indices in self.dag_levels:
            wave     = [self.merge_order[i] for i in level_indices]
            d_tokens = gpu_apply_wave(d_tokens, wave, self.vocab_cap)

        return d_tokens.get().tolist() if CUDA_AVAILABLE else d_tokens.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    # ── Decoding ──────────────────────────────────────────────────────────

    def decode(self, token_ids: List[int]) -> str:
        out = bytearray()
        for tid in token_ids:
            self._expand(int(tid), out)
        return out.decode('utf-8', errors='replace')

    def _expand(self, tid: int, out: bytearray):
        if tid < 256:
            out.append(tid)
        elif tid in self.id_to_merge:
            a, b = self.id_to_merge[tid]
            self._expand(a, out)
            self._expand(b, out)

    # ── Info ──────────────────────────────────────────────────────────────

    def dag_depth(self) -> int:
        return len(self.dag_levels)

    def parallelism_pct(self) -> float:
        n = len(self.merge_order)
        if n == 0:
            return 0.0
        return (1 - self.dag_depth() / n) * 100