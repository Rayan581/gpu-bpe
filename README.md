GPU-Accelerated Parallel BPE Tokenization in Distributed LLM Training
=====================================================================

Status: research prototype, not production-ready
Last reviewed: May 2026

This repository explores a training pipeline that combines byte-level BPE
tokenization, digit-span guarding for math text, adaptive gradient
quantization, and a simulated parameter-server training architecture.

The codebase is useful as a prototype and as a project scaffold, but several
claims in the paper/docs are ahead of the current implementation. In
particular, the full distributed path currently needs fixes before it can be
treated as a reliable experiment runner.

What Is Implemented
-------------------

- Byte-level BPE training and encoding in `tokenizer/`.
- A `GPUBPETokenizer` wrapper that can place padded encoded batches on CUDA.
- A `SemanticGuardedTokenizer` wrapper that marks digit spans before encoding
  and removes those markers after decoding.
- Synthetic OpenWebText-like, GSM8K-like, and MATH-like data generation in
  `utils/data.py`.
- Baseline local training in `scripts/train_baseline.py`.
- Prototype distributed components in `dist/`:
  - `ControlLayer`
  - `ParameterServer`
  - `TrainingWorker`
- Prototype INT8/INT4 quantization in `compression/adaptive_quant.py`.
- Evaluation scripts for synthetic GSM8K-style answer matching and perplexity.
- A Colab notebook, `train_notebook.ipynb`, that uses real Hugging Face
  OpenWebText/GSM8K loading and Google Drive caching.
- A draft IEEE-style paper in `gpu_bpe_paper.tex`.

Important Limitations
---------------------

Read this section before using the project for reported results.

1. `run_experiment.py` currently passes unsupported CLI flags such as
   `--learning_rate` to training scripts and does not fail when a child script
   exits with an error. Prefer running scripts directly for now.

2. The full distributed trainer is currently broken at runtime:
   `TrainingWorker.push_gradients()` uses `self.step`, while the worker state is
   stored as `self.train_steps`.

3. The tokenizer is not a true vectorized GPU BPE merge implementation. The
   merge logic runs through Python lists; CUDA is mainly used for tensor staging
   after tokenization.

4. HSG does not create atomic digit tokens. It inserts marker strings such as
   `<|digit_start|>` and `<|digit_end|>` before byte-level tokenization, then
   strips them on decode. This may help preserve boundaries in text form, but it
   does not guarantee that numbers remain single tokens.

5. The ablation scripts mostly log simulated values. They do not currently
   perform a full training/evaluation loop that supports the tables in the
   paper.

6. The parameter server applies plain gradient descent, not Adam, and staleness
   scaling is commented out. The current code should be described as a
   simulated parameter-server prototype, not a complete DC-ASGD implementation.

7. Evaluation scripts use synthetic data by default and simple answer matching.
   They are smoke tests, not rigorous GSM8K or language-model evaluations.

8. `gpu_bpe_paper.tex` is a draft. It should be revised to match the current
   implementation before submission.

Repository Layout
-----------------

```text
.
|-- README.md
|-- QUICKSTART.md
|-- GPU_SETUP_GUIDE.md
|-- DOCUMENTATION_STRUCTURE.txt
|-- requirements.txt
|-- run_experiment.py
|-- run_experiments.sh
|-- train_notebook.ipynb
|-- gpu_bpe_paper.tex
|-- tokenizer/
|   |-- bpe.py
|   |-- gpu_bpe.py
|   |-- hsg.py
|   |-- sequential_bpe.py
|   `-- __init__.py
|-- compression/
|   |-- adaptive_quant.py
|   `-- __init__.py
|-- dist/
|   |-- control.py
|   |-- parameter_server.py
|   |-- worker.py
|   `-- __init__.py
|-- scripts/
|   |-- train_baseline.py
|   |-- train_full.py
|   |-- ablation_a.py
|   |-- ablation_b.py
|   |-- eval_gsm8k.py
|   `-- eval_perplexity.py
`-- utils/
    |-- data.py
    |-- metrics.py
    `-- __init__.py
```

Setup
-----

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Core Components
---------------

### Tokenizer

`tokenizer/bpe.py` contains the speculative BPE merge-selection logic and wave
application helpers. `tokenizer/gpu_bpe.py` wraps that logic in a byte-level
tokenizer API.

Example:

```python
from tokenizer.gpu_bpe import GPUBPETokenizer

tokenizer = GPUBPETokenizer(vocab_size=50257, use_gpu=True)
ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)
```

Batch encoding returns padded tensors and attention masks:

```python
token_ids, attention_mask = tokenizer.encode_batch(
    ["hello world", "another example"],
    max_length=32,
)
```

### Hybrid Semantic Guard

`tokenizer/hsg.py` wraps digit spans with textual markers before tokenization:

```python
from tokenizer.hsg import SemanticGuardedTokenizer

guarded = SemanticGuardedTokenizer(tokenizer, enable_hsg=True)
ids = guarded.encode("There are 42 apples.")
decoded = guarded.decode(ids)
```

Current behavior: HSG preserves the original decoded text by adding and removing
markers. It does not yet add protected special-token IDs to the BPE vocabulary.

### Data Utilities

`utils/data.py` generates synthetic datasets:

- `owt`: OpenWebText-like synthetic documents
- `gsm8k`: GSM8K-like arithmetic word problems
- `math`: simple synthetic algebra/geometry problems
- `synthetic`: repeated toy text

Example:

```python
from utils.data import get_dataloader

loader = get_dataloader("owt", tokenizer, batch_size=8, num_docs=100)
```

### Compression

`compression/adaptive_quant.py` provides an experimental INT8/INT4 quantizer.
It reports theoretical compression ratios of 4x for INT8 and 8x for INT4.

The implementation needs more work before it should be used as evidence for
convergence or accuracy claims.

### Distributed Prototype

The `dist/` package contains a single-machine simulation of:

- scheduling and staleness tracking
- parameter pull/push
- worker-side gradient computation

It is not a real multi-machine RPC/NCCL setup.

Recommended Commands
--------------------

### Component smoke tests

These are the safest first checks:

```bash
python tokenizer/hsg.py
python compression/adaptive_quant.py
python dist/control.py
python dist/parameter_server.py
python utils/metrics.py
```

`python tokenizer/gpu_bpe.py` can also be used for tokenizer experimentation,
but treat timings as local prototype behavior.

### Baseline training

This is the most usable training script in the repository:

```bash
python scripts/train_baseline.py ^
  --num_steps 5 ^
  --batch_size 4 ^
  --max_length 128 ^
  --save_dir outputs/baseline
```

On Bash:

```bash
python scripts/train_baseline.py \
  --num_steps 5 \
  --batch_size 4 \
  --max_length 128 \
  --save_dir outputs/baseline
```

The script writes `metrics.json` to the selected output directory.

### Full training prototype

This command documents the intended interface, but the current implementation
has the `TrainingWorker.self.step` bug noted above:

```bash
python scripts/train_full.py \
  --num_steps 5 \
  --batch_size 4 \
  --max_length 128 \
  --num_workers 2 \
  --enable_hsg True \
  --save_dir outputs/full
```

### Ablation scripts

These scripts currently log simulated losses and should be treated as scaffolds:

```bash
python scripts/ablation_a.py --num_steps 5 --batch_size 4
python scripts/ablation_b.py --num_steps 5 --batch_size 4
```

### Evaluation smoke tests

These run on synthetic datasets unless changed:

```bash
python scripts/eval_gsm8k.py --num_problems 10 --save_dir outputs/eval
python scripts/eval_perplexity.py --num_docs 10 --save_dir outputs/eval
```

### Colab notebook

Use `train_notebook.ipynb` for the Colab workflow. It mounts Google Drive and
uses:

```text
/content/drive/MyDrive/PDC/cache
```

for checkpoints, logs, results, and dataset caches.

If Colab keeps downloading OpenWebText/GSM8K again, inspect:

```text
/content/drive/MyDrive/PDC/cache/datasets
```

The saved Hugging Face dataset folders must contain files such as `state.json`,
`dataset_info.json`, and Arrow data files. Empty folders are not valid caches.

Known Issues To Fix Next
------------------------

High priority:

- Replace `self.step` with `self.train_steps` in `dist/worker.py`.
- Make `run_experiment.py` use only supported flags and call
  `subprocess.run(..., check=True)`.
- Decide whether `train_full.py` should use true worker steps or remain a
  simulation, then update names/docs accordingly.
- Make the paper match the implementation, especially GPU-tokenization,
  GPT-2, HSG, DC-ASGD, and ablation claims.

Medium priority:

- Add real special-token handling for HSG if digit spans must be protected.
- Fix or redesign error-feedback handling in `AdaptiveQuantizer`.
- Add real checkpoint saving/loading for training scripts.
- Add deterministic seeds for synthetic data generation.
- Fix `TextDataset` so short or exactly `max_length` documents still produce
  examples where appropriate.
- Add tests that assert command-line examples in this README actually run.

Paper Notes
-----------

`gpu_bpe_paper.tex` is currently a draft, not a verified report. Before using it
for submission:

- Replace unsupported performance numbers with measured results.
- Correct the description of tokenizer execution.
- Correct the description of HSG.
- Correct the model description if the code remains `nn.TransformerEncoderLayer`
  rather than GPT-2.
- Verify all citations and arXiv IDs.
- Compile the paper with a local LaTeX toolchain and inspect warnings.

Development Notes
-----------------

The project is best viewed as a collection of research components rather than a
finished training framework. A good development order is:

1. Make the script runner fail correctly.
2. Fix the distributed worker runtime error.
3. Add a small automated smoke-test suite.
4. Measure baseline behavior on the same hardware and data.
5. Only then update the paper with real experimental numbers.

License
-------

No explicit license file is currently present in the workspace. Add one before
publishing or sharing the project as open source.
