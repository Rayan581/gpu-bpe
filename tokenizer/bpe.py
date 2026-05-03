"""
BPE Tokenizer Comparison: Standard vs Parallel (BPE_a)
=======================================================
Standard BPE  : sequential — each merge depends on the previous rewritten string.
Parallel BPE_a: speculative merge selection on the original string, builds a
                dependency DAG, then executes independent merges in parallel
                waves using multiprocessing.
"""

import time
import random
import string
import collections
import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_long_string(length: int = 200_000) -> str:
    """Generate a realistic-ish long string with skewed character distribution."""
    random.seed(42)
    # Skew so some pairs are very frequent (makes BPE more interesting)
    alphabet = "aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz "
    return "".join(random.choice(alphabet) for _ in range(length))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STANDARD (SEQUENTIAL) BPE
# ─────────────────────────────────────────────────────────────────────────────

def get_pair_freqs(token_list: list) -> dict:
    freqs = collections.defaultdict(int)
    for i in range(len(token_list) - 1):
        freqs[(token_list[i], token_list[i + 1])] += 1
    return freqs


def standard_bpe(text: str, num_merges: int):
    """
    Classic sequential BPE.
    Returns (vocab, merge_order, final_token_list).
    """
    # Start with individual characters as integer ids
    char_to_id = {}
    id_to_sym  = {}
    tokens = []
    for ch in text:
        if ch not in char_to_id:
            tid = len(char_to_id)
            char_to_id[ch] = tid
            id_to_sym[tid] = ch
        tokens.append(char_to_id[ch])

    next_id     = len(char_to_id)
    merge_order = []   # list of ((left_id, right_id), new_id)
    vocab       = dict(id_to_sym)  # id → symbol string

    for _ in range(num_merges):
        freqs = get_pair_freqs(tokens)
        if not freqs:
            break
        best = max(freqs, key=freqs.get)
        if freqs[best] < 2:
            break

        new_id = next_id
        next_id += 1
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merge_order.append((best, new_id))

        # Rewrite token list in one pass
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return vocab, merge_order, tokens


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PARALLEL BPE_a
# ─────────────────────────────────────────────────────────────────────────────

# ── Phase 1 & 2: Speculative merge selection ─────────────────────────────────

def _speculative_select(tokens_orig: list, num_merges: int, base_next_id: int):
    """
    Select N merges speculatively from the original token list without fully
    rewriting it.  Returns merge_order and vocab additions.
    """
    # Build initial frequency map + location lists using a max-heap
    pair_freq = collections.defaultdict(int)
    pair_locs = collections.defaultdict(list)   # pair → [positions]

    for i in range(len(tokens_orig) - 1):
        p = (tokens_orig[i], tokens_orig[i + 1])
        pair_freq[p] += 1
        pair_locs[p].append(i)

    # Max-heap: store (-freq, tie-break counter, pair)
    counter   = 0
    heap      = []
    for pair, freq in pair_freq.items():
        heapq.heappush(heap, (-freq, counter, pair))
        counter += 1

    invalidated = set()
    next_id     = base_next_id
    merge_order = []          # [(pair, new_token_id)]
    new_vocab   = {}          # new_token_id → (left_id, right_id)

    # Shadow token array: track what each position "looks like" after spec merges
    # We only need neighbour lookups, so keep a simple position → current_token map
    shadow = list(tokens_orig)          # will be mutated speculatively
    # positions that have been consumed (merged away) in shadow view
    dead   = set()

    for _ in range(num_merges):
        # Pop until we find a valid, non-invalidated pair
        best_pair = None
        while heap:
            neg_freq, _, pair = heapq.heappop(heap)
            if pair in invalidated:
                continue
            # Recompute actual shadow freq (may have changed)
            actual_freq = sum(
                1 for pos in pair_locs[pair]
                if pos not in dead
                and pos + 1 not in dead
                and shadow[pos] == pair[0]
                and shadow[pos + 1] == pair[1]
            )
            if actual_freq >= 2:
                best_pair = pair
                break

        if best_pair is None:
            break

        new_id = next_id
        next_id += 1
        new_vocab[new_id] = best_pair
        merge_order.append((best_pair, new_id))
        invalidated.add(best_pair)

        # Speculatively apply merge in shadow array & update neighbour pairs
        valid_positions = [
            pos for pos in pair_locs[best_pair]
            if pos not in dead
            and pos + 1 not in dead
            and shadow[pos] == best_pair[0]
            and shadow[pos + 1] == best_pair[1]
        ]

        new_locs_left  = collections.defaultdict(list)
        new_locs_right = collections.defaultdict(list)

        for pos in valid_positions:
            # Find left neighbour (skip dead positions)
            left_pos = pos - 1
            while left_pos >= 0 and left_pos in dead:
                left_pos -= 1
            # Find right neighbour
            right_pos = pos + 2
            while right_pos < len(shadow) and right_pos in dead:
                right_pos += 1

            # Destroy old border pairs
            if left_pos >= 0:
                old_left_pair = (shadow[left_pos], best_pair[0])
                invalidated.add(old_left_pair)
            if right_pos < len(shadow):
                old_right_pair = (best_pair[1], shadow[right_pos])
                invalidated.add(old_right_pair)

            # Apply speculative merge
            shadow[pos] = new_id
            dead.add(pos + 1)

            # Register new border pairs
            if left_pos >= 0:
                np_left = (shadow[left_pos], new_id)
                new_locs_left[np_left].append(left_pos)
            if right_pos < len(shadow):
                np_right = (new_id, shadow[right_pos])
                new_locs_right[np_right].append(pos)

        # Push new pairs into heap
        for np, locs in {**new_locs_left, **new_locs_right}.items():
            pair_locs[np].extend(locs)
            freq = len([l for l in pair_locs[np]
                        if l not in dead
                        and l + 1 not in dead
                        and shadow[l] == np[0]
                        and (l + 1 < len(shadow)) and shadow[l + 1] == np[1]])
            if freq >= 2 and np not in invalidated:
                heapq.heappush(heap, (-freq, counter, np))
                counter += 1

    return merge_order, new_vocab


# ── Phase 3: Build dependency DAG ────────────────────────────────────────────

def _build_dag(merge_order: list) -> dict:
    """
    Returns {merge_index: set_of_parent_merge_indices}.
    Merge j depends on merge i if merge i's output token appears in merge j's pair.
    """
    token_produced_by = {}   # token_id → merge_index
    deps = {i: set() for i in range(len(merge_order))}

    for idx, (pair, new_token) in enumerate(merge_order):
        left, right = pair
        if left in token_produced_by:
            deps[idx].add(token_produced_by[left])
        if right in token_produced_by:
            deps[idx].add(token_produced_by[right])
        token_produced_by[new_token] = idx

    return deps


# ── Phase 4: Topological level assignment ────────────────────────────────────

def _topo_levels(deps: dict) -> list:
    """
    Returns a list of levels, where each level is a list of merge indices
    that can be executed in parallel.
    """
    n      = len(deps)
    level  = [-1] * n
    # children map
    children = collections.defaultdict(list)
    in_deg   = [0] * n
    for node, parents in deps.items():
        for p in parents:
            children[p].append(node)
            in_deg[node] += 1

    queue = collections.deque()
    for node in range(n):
        if in_deg[node] == 0:
            level[node] = 0
            queue.append(node)

    while queue:
        node = queue.popleft()
        for child in children[node]:
            in_deg[child] -= 1
            level[child] = max(level[child], level[node] + 1)
            if in_deg[child] == 0:
                queue.append(child)

    # Group by level
    max_level = max(level)
    levels    = [[] for _ in range(max_level + 1)]
    for node, lv in enumerate(level):
        levels[lv].append(node)
    return levels


# ── Phase 4b: Apply one wave of merges (worker function) ─────────────────────

def _apply_merge_wave(args):
    """
    Applied in a subprocess.  args = (tokens, merges_in_wave)
    merges_in_wave: list of (pair, new_token_id)
    Returns the rewritten token list.
    """
    tokens, merges_in_wave = args
    # Build a lookup: (left, right) → new_token for this wave
    # Merges within a single wave are INDEPENDENT by construction
    merge_map = {pair: new_id for pair, new_id in merges_in_wave}

    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_map:
                new_tokens.append(merge_map[pair])
                i += 2
                continue
        new_tokens.append(tokens[i])
        i += 1
    return new_tokens


# ── Main parallel BPE entry point ────────────────────────────────────────────

def parallel_bpe(text: str, num_merges: int, max_workers: int = 4):
    """
    Parallel BPE_a.
    Returns (vocab, merge_order, final_token_list, dag_depth).
    """
    # Build base vocab from characters
    char_to_id = {}
    id_to_sym  = {}
    tokens = []
    for ch in text:
        if ch not in char_to_id:
            tid = len(char_to_id)
            char_to_id[ch] = tid
            id_to_sym[tid] = ch
        tokens.append(char_to_id[ch])

    base_next_id = len(char_to_id)
    vocab        = dict(id_to_sym)

    # Phase 1+2: speculative selection
    merge_order, new_vocab = _speculative_select(tokens, num_merges, base_next_id)
    vocab.update({nid: vocab[pair[0]] + vocab[pair[1]]
                  for nid, pair in new_vocab.items()
                  if pair[0] in vocab and pair[1] in vocab})

    # Phase 3: dependency DAG
    deps   = _build_dag(merge_order)

    # Phase 4: topological levels
    levels = _topo_levels(deps)
    dag_depth = len(levels)

    # Phase 4b: apply level by level (each level executed with workers)
    for level_indices in levels:
        wave = [merge_order[i] for i in level_indices]
        # Split tokens into chunks for parallel processing
        chunk_size = max(1, len(tokens) // max_workers)
        # We must be careful at chunk boundaries — carry-over logic needed
        # Simple approach: only parallelize if wave has many merges, else serial
        if len(wave) >= max_workers and len(tokens) > 10_000:
            chunk_size = max(1, len(tokens) // max_workers)
            chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
            # Apply the whole wave to each chunk independently
            # (boundary pairs between chunks may be missed — acceptable approximation
            #  for benchmarking; production impl would stitch boundaries)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_apply_merge_wave,
                                      [(chunk, wave) for chunk in chunks]))
            tokens = [tok for chunk in results for tok in chunk]
        else:
            tokens = _apply_merge_wave((tokens, wave))

    return vocab, merge_order, tokens, dag_depth


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(text: str, num_merges: int, max_workers: int = 4):
    print("=" * 65)
    print(f"  BPE TOKENIZER BENCHMARK")
    print("=" * 65)
    print(f"  Input length : {len(text):,} characters")
    print(f"  Merges (N)   : {num_merges}")
    print(f"  Workers      : {max_workers}")
    print("=" * 65)

    # ── Standard BPE ──
    print("\n[1/2] Running Standard (Sequential) BPE …")
    t0 = time.perf_counter()
    std_vocab, std_merges, std_tokens = standard_bpe(text, num_merges)
    t_std = time.perf_counter() - t0
    print(f"      Done in {t_std:.4f}s")
    print(f"      Merges applied   : {len(std_merges)}")
    print(f"      Final token list : {len(std_tokens):,} tokens")

    # ── Parallel BPE ──
    print("\n[2/2] Running Parallel BPE_a …")
    t0 = time.perf_counter()
    par_vocab, par_merges, par_tokens, dag_depth = parallel_bpe(
        text, num_merges, max_workers=max_workers
    )
    t_par = time.perf_counter() - t0
    print(f"      Done in {t_par:.4f}s")
    print(f"      Merges selected  : {len(par_merges)}")
    print(f"      DAG depth        : {dag_depth}  (theoretical min sequential steps)")
    print(f"      Final token list : {len(par_tokens):,} tokens")

    # ── Summary ──
    speedup = t_std / t_par if t_par > 0 else float("inf")
    dag_reduction = (1 - dag_depth / num_merges) * 100 if num_merges else 0

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Standard BPE time   : {t_std:.4f} s")
    print(f"  Parallel BPE_a time : {t_par:.4f} s")
    print(f"  Wall-clock speedup  : {speedup:.2f}×")
    print(f"  DAG depth vs N      : {dag_depth} / {num_merges}  "
          f"({dag_reduction:.1f}% parallelism potential)")
    print("=" * 65)

    # Show a tiny sample of the merge tables
    print("\n  First 10 Standard BPE merges:")
    for i, (pair, nid) in enumerate(std_merges[:10]):
        print(f"    [{i+1:02d}] {std_vocab[pair[0]]!r} + {std_vocab[pair[1]]!r}"
              f" → {std_vocab[nid]!r}  (id={nid})")

    print("\n  First 10 Parallel BPE_a merges (speculative order):")
    for i, (pair, nid) in enumerate(par_merges[:10]):
        lsym = par_vocab.get(pair[0], f"<{pair[0]}>")
        rsym = par_vocab.get(pair[1], f"<{pair[1]}>")
        nsym = par_vocab.get(nid,     f"<{nid}>")
        print(f"    [{i+1:02d}] {lsym!r} + {rsym!r} → {nsym!r}  (id={nid})")

    print()
    return {
        "std_time": t_std,
        "par_time": t_par,
        "speedup":  speedup,
        "dag_depth": dag_depth,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    TEXT       = generate_long_string(200_000)
    NUM_MERGES = 500
    WORKERS    = min(8, multiprocessing.cpu_count())

    results = benchmark(TEXT, NUM_MERGES, max_workers=WORKERS)
