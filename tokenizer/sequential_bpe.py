"""
Sequential (Standard) BPE Tokenizer

A simple, byte-level BPE implementation that performs merges sequentially.
This serves as the baseline for comparing against GPU BPE's parallel approach.

Algorithm:
  1. Start with individual bytes (0-255) as tokens
  2. For each of N merge iterations:
     - Count all adjacent token pair frequencies
     - Select the most frequent pair
     - Create a new token ID for the merged pair
     - Rewrite the token sequence (single pass)
  3. Encode new text by:
     - Convert bytes to initial token list
     - Apply learned merges in order

This is deterministic and simple — perfect for a clean baseline.
"""

from typing import List, Dict, Tuple, Optional
import collections


class SequentialBPETokenizer:
    """
    Standard sequential byte-level BPE tokenizer.

    Training complexity: O(num_merges * corpus_size)
    Encoding complexity: O(text_length * num_merges)  [in worst case]

    Simple, deterministic, no parallelization.
    """

    def __init__(self, vocab_size: int = 50257):
        """
        Args:
            vocab_size: Target vocabulary size (includes 256 base bytes + merges)
        """
        self.vocab_size = vocab_size
        self.max_merges = vocab_size - 256

        # Merge vocabulary: pair → token_id
        # Example: (101, 102) -> 256  means bytes 101 and 102 merge into token 256
        self.merges: Dict[Tuple[int, int], int] = {}

        # Reverse mapping for decoding: token_id → (left_id, right_id)
        self.id_to_pair: Dict[int, Tuple[int, int]] = {}

        # Token ID → byte string for final decoding
        self._token_to_bytes: Dict[int, bytes] = {}

        # Track merge order for reproducibility
        self.merge_order: List[Tuple[int, int]] = []

    def train(self, texts: List[str], num_merges: Optional[int] = None) -> int:
        """
        Train the tokenizer on a list of texts.

        Args:
            texts: List of training texts
            num_merges: Number of merges to perform (default: vocab_size - 256)

        Returns:
            Actual number of merges performed
        """
        if num_merges is None:
            num_merges = self.max_merges
        else:
            num_merges = min(num_merges, self.max_merges)

        # Concatenate corpus with newline separators between documents
        # This matches GPU BPE's behavior
        corpus = '\n'.join(texts)
        tokens = list(corpus.encode('utf-8'))

        if len(tokens) < 2:
            return 0

        print(
            f"  SequentialBPE: {len(tokens):,} bytes, target {num_merges} merges")

        from tokenizer._progress import get_tqdm as _get_tqdm
        _tqdm = _get_tqdm()

        import time as _time
        t_start = _time.perf_counter()

        pbar = (
            _tqdm(total=num_merges, desc="Sequential BPE", unit="merge",
                  dynamic_ncols=True)
            if _tqdm is not None else None
        )

        next_token_id = 256
        merges_done = 0

        # ── Build initial data structures ────────────────────────────────
        # pair_freqs   : pair -> current frequency (kept incrementally)
        # pair_locs    : pair -> set of positions where it appears
        # prev / next_ : doubly-linked list over the token array so we can
        #                do O(1) neighbour lookups without shifting a list
        n = len(tokens)
        prev = list(range(-1, n - 1))   # prev[i] = previous alive index
        # next_[i] = next alive index (n = end)
        next_ = list(range(1, n + 1))
        alive = [True] * n

        pair_freqs: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        pair_locs:  Dict[Tuple[int, int], set] = collections.defaultdict(set)

        for i in range(n - 1):
            p = (tokens[i], tokens[i + 1])
            pair_freqs[p] += 1
            pair_locs[p].add(i)

        # Max-heap via negated frequencies
        import heapq as _heapq
        heap = [(-freq, pair) for pair, freq in pair_freqs.items()]
        _heapq.heapify(heap)

        for merge_idx in range(num_merges):
            t0 = _time.perf_counter()

            # Pop until we find a pair whose heap entry is still current
            best_pair = None
            freq = 0
            while heap:
                neg_freq, p = _heapq.heappop(heap)
                current = pair_freqs.get(p, 0)
                if current >= 2 and current == -neg_freq:
                    best_pair = p
                    freq = current
                    break

            if best_pair is None:
                break

            # Register the merge
            new_token_id = next_token_id
            self.merges[best_pair] = new_token_id
            self.id_to_pair[new_token_id] = best_pair
            self.merge_order.append(best_pair)
            next_token_id += 1
            merges_done += 1

            # ── Apply merge incrementally ────────────────────────────────
            positions = list(pair_locs.pop(best_pair, set()))
            pair_freqs.pop(best_pair, None)
            toks_remaining = n - sum(1 for a in alive if not a)

            for pos in positions:
                # Skip if either slot was already consumed
                if not alive[pos]:
                    continue
                right = next_[pos]
                if right >= n or not alive[right]:
                    continue
                if tokens[pos] != best_pair[0] or tokens[right] != best_pair[1]:
                    continue

                # --- remove left neighbour pair ---
                lft = prev[pos]
                if lft >= 0 and alive[lft]:
                    lp = (tokens[lft], tokens[pos])
                    pair_freqs[lp] -= 1
                    pair_locs[lp].discard(lft)

                # --- remove right neighbour pair ---
                rgt = next_[right]
                if rgt < n and alive[rgt]:
                    rp = (tokens[right], tokens[rgt])
                    pair_freqs[rp] -= 1
                    pair_locs[rp].discard(right)

                # --- perform the merge ---
                tokens[pos] = new_token_id
                alive[right] = False
                # stitch linked list: skip `right`
                next_[pos] = next_[right]
                if next_[right] < n:
                    prev[next_[right]] = pos

                # --- add new left neighbour pair ---
                if lft >= 0 and alive[lft]:
                    lp2 = (tokens[lft], new_token_id)
                    pair_freqs[lp2] += 1
                    pair_locs[lp2].add(lft)
                    _heapq.heappush(heap, (-pair_freqs[lp2], lp2))

                # --- add new right neighbour pair ---
                rgt2 = next_[pos]
                if rgt2 < n and alive[rgt2]:
                    rp2 = (new_token_id, tokens[rgt2])
                    pair_freqs[rp2] += 1
                    pair_locs[rp2].add(pos)
                    _heapq.heappush(heap, (-pair_freqs[rp2], rp2))

            step_ms = (_time.perf_counter() - t0) * 1000
            toks_remaining = sum(alive)
            if pbar is not None:
                pbar.set_postfix({
                    "merge":    merges_done,
                    "freq":     freq,
                    "toks":     toks_remaining,
                    "ms/merge": f"{step_ms:.1f}",
                })
                pbar.update(1)

        # Rebuild compact token list from linked-list structure
        tokens = [tokens[i] for i in range(n) if alive[i]]

        if pbar is not None:
            pbar.close()

        elapsed = _time.perf_counter() - t_start
        print(f"  SequentialBPE done: {merges_done} merges in {elapsed:.2f}s "
              f"({elapsed / max(merges_done, 1) * 1000:.1f} ms/merge avg)")

        # Build token_to_bytes for decoding
        self._build_token_to_bytes()

        return merges_done

    @staticmethod
    def _get_pair_freqs(tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """Count frequencies of all adjacent pairs in token sequence."""
        freqs = collections.defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            freqs[pair] += 1
        return freqs

    @staticmethod
    def _apply_merge(
        tokens: List[int],
        pair: Tuple[int, int],
        new_token_id: int
    ) -> List[int]:
        """Apply a single merge to the token sequence (single pass)."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _build_token_to_bytes(self):
        """Build a cache mapping token_id -> bytes for fast decoding."""
        self._token_to_bytes.clear()

        # Base tokens (0-255) map directly to bytes
        for byte_val in range(256):
            self._token_to_bytes[byte_val] = bytes([byte_val])

        # Merged tokens are built from their component pairs
        for new_id, (left_id, right_id) in self.id_to_pair.items():
            left_bytes = self._token_to_bytes.get(left_id, b'')
            right_bytes = self._token_to_bytes.get(right_id, b'')
            self._token_to_bytes[new_id] = left_bytes + right_bytes

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Start with individual bytes
        tokens = list(text.encode('utf-8'))

        # Apply each learned merge in order
        for pair, new_id in zip(self.merge_order,
                                [self.merges[p] for p in self.merge_order]):
            tokens = self._apply_merge(tokens, pair, new_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back into text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        # Get bytes for each token and concatenate
        byte_sequences = []
        for token_id in token_ids:
            if token_id in self._token_to_bytes:
                byte_sequences.append(self._token_to_bytes[token_id])
            else:
                # Unknown token — replace with placeholder
                byte_sequences.append(b'?')

        # Concatenate and decode, replacing invalid UTF-8
        full_bytes = b''.join(byte_sequences)
        return full_bytes.decode('utf-8', errors='replace')

    def __repr__(self):
        return (
            f"SequentialBPETokenizer(vocab_size={self.vocab_size}, "
            f"merges_trained={len(self.merges)})"
        )
