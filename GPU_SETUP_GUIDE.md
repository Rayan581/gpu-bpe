# GPU Setup Guide - Quick Reference

## 🚀 Quick Start (Choose Your GPU)

### Colab T4 (16GB) - RECOMMENDED FOR FULL DATASET
```
1. Open: https://colab.research.google.com
2. File > Upload notebook > train_notebook.ipynb
3. Runtime > Change runtime type > GPU (T4)
4. Run cells in order
5. Checkpoints auto-save to Google Drive
```

**Configuration (already set for full dataset)**:
- Batch size: 8
- Hidden size: 256
- Num layers: 4
- Max length: 256
- **Memory usage**: ~12-14 GB
- **Training time**: ~4-6 hours for full OpenWebText

---

### RTX 4050 Laptop (8GB) - LOCAL MACHINE
```
1. Install: pip install -r requirements.txt
2. Start Jupyter: jupyter notebook
3. Open: train_notebook.ipynb
4. Runtime > Select Kernel > Python 3
5. Run cells in order (Shift+Enter)
6. Checkpoints auto-save locally
```

**Configuration (optimized for full dataset)**:
- Batch size: 4-8
- Hidden size: 256
- Num layers: 4
- Max length: 256
- **Memory usage**: ~6-7 GB
- **Training time**: ~6-10 hours for full OpenWebText

**If you get OOM error**:
```python
# In Step 7, modify config:
config.batch_size = 4           # ← reduce batch
config.hidden_size = 128        # ← reduce model size
config.max_length = 128         # ← reduce sequence length
config.num_layers = 2           # ← fewer layers
```

---

### Other GPUs

| GPU | Memory | Batch Size | Max Length | Status |
|-----|--------|-----------|-----------|--------|
| A100 (80GB) | 80GB | 32 | 2048 | ✓ Excellent |
| A10 (24GB) | 24GB | 16 | 512 | ✓ Good |
| RTX 3090 (24GB) | 24GB | 16 | 512 | ✓ Good |
| RTX 4050 | 8GB | 4-8 | 256 | ✓ Works |
| RTX 3060 (12GB) | 12GB | 8 | 256 | ✓ Good |
| Tesla T4 (Colab) | 16GB | 8 | 256 | ✓ Recommended |
| M1/M2 GPU (Mac) | 8GB | 4 | 128 | ⚠ Slow |
| CPU | RAM | 1-2 | 128 | ✗ Not recommended |

---

## 📊 Hardware Comparison (Full Dataset)

### Performance Comparison (Full OpenWebText - 1 epoch)

```
Colab T4:     ████████████░░░░░░░░ 5 hours
RTX 4050:     ████████████████░░░░ 8 hours
RTX 3090:     ███░░░░░░░░░░░░░░░░ 2.5 hours
A100:         █░░░░░░░░░░░░░░░░░░ 1 hour
CPU:          ████████████████████ 20+ hours
```

### Memory Usage (Training on Full Dataset)

```
Colab T4 (16GB):   ████████████░░░░░░░░ 12-14 GB used
RTX 4050 (8GB):    ██████░░░░░░░░░░░░░░ 6-7 GB used
RTX 3090 (24GB):   █████████░░░░░░░░░░░ 9-10 GB used
RTX 2060 (6GB):    ██████░░░░░░░░░░░░░░ 5.5-6 GB used
```

---

## ⚡ Optimization Tips by Hardware

### For RTX 4050 (8GB) - Get More Speed
```python
# Option 1: Faster but less accurate
config.batch_size = 8           # Max safe batch
config.hidden_size = 256        # Keep decent model
config.max_length = 256         # Standard length
config.num_layers = 4           # Standard depth

# Option 2: Smaller model, faster
config.batch_size = 8
config.hidden_size = 128        # Smaller
config.num_layers = 2           # Fewer layers
config.max_length = 256

# Option 3: Best accuracy
config.batch_size = 4           # Slower but better
config.hidden_size = 256
config.num_layers = 4
config.max_length = 512         # Longer sequences
```

### For Colab T4 (16GB) - Maximum Training
```python
# Use defaults - well tuned for T4
config.batch_size = 8
config.hidden_size = 256
config.num_layers = 4
config.max_length = 256

# Optional: increase for better results
config.batch_size = 16
config.hidden_size = 512
config.max_length = 512
config.num_layers = 6
```

### For High-End GPUs (A10, RTX 3090+)
```python
# Go bigger for better model
config.batch_size = 32
config.hidden_size = 768
config.num_layers = 12
config.max_length = 1024
```

---

## 🔧 Common Issues & Solutions

### Error: "CUDA out of memory"
```python
# Immediate fix:
config.batch_size = 4  # Reduce by half

# If still OOM:
config.hidden_size = 128
config.max_length = 128

# Nuclear option:
config.num_layers = 2
config.hidden_size = 64
```

### Colab Runtime Disconnects
```
✓ Good news: Checkpoints save every 100 steps
✓ Resume: Just re-run and choose "Resume from checkpoint"
✓ No training lost beyond last checkpoint

Typical: Save ~5-10 checkpoints per session
```

### Slow Training on RTX 4050
```
Check: GPU utilization in logs (Step 8)

If <50% utilized:
  → Problem is data loading
  → Solution: Use fewer workers (already 0 in notebook)

If >90% utilized:
  → Problem is batch size too small
  → Solution: Increase batch_size if memory allows
```

### Training Loss Not Decreasing
```
1. Check learning rate (try 1e-4 to 5e-5)
2. Add warmup: config.warmup_steps = 500
3. Check data: Print sample batch
4. Try smaller batch size (stabilizes training)
5. Check gradients: print(grad.norm()) in training loop
```

---

## 📈 Expected Results (Full Dataset)

### Training Loss (Full OpenWebText, 1 epoch)
```
Step 0:     2.3 (random predictions)
25% data:   1.8 (22% reduction)
50% data:   1.4 (39% reduction)
75% data:   1.0 (57% reduction)
100% data:  0.5-0.8 (65-78% final reduction)
```

### GSM8K Accuracy (Full 7,473 examples)
```
Baseline:       ~15-20% (random)
After training: ~30-45% (improved)
With HSG:       ~35-50% (digit preservation)
Multi-epoch:    Could reach 60%+
```

### GPU Memory Over Training
```
Start:              ~0 GB
After first batch:  ~5-6 GB (RTX 4050)
Peak:               ~6-7 GB
Stable during:      6-7 GB for entire epoch
After training:     Clears via torch.cuda.empty_cache()
```

### Training Throughput
```
Colab T4:    ~500-700 tokens/sec
RTX 4050:    ~300-400 tokens/sec
RTX 3090:    ~1000-1200 tokens/sec
```

---

## 🎯 Recommended Setup (Full Dataset)

### If you have RTX 4050 + Colab:
```
1. Use Colab FIRST (4-6 hours, handles 20GB dataset)
2. Download results and checkpoints
3. Run locally only if you need more control
4. Local training: 6-10 hours on RTX 4050
```

### If only RTX 4050:
```
1. Use batch_size = 4-8 for full dataset
2. Run overnight (6-10 hours)
3. Save checkpoints to external drive (backup)
4. Monitor GPU in Task Manager
5. Resume from checkpoint if interrupted
```

### If you have Colab access (RECOMMENDED):
```
1. Use Colab T4 for full dataset training
2. Training: 4-6 hours (shorter than local RTX 4050)
3. Checkpoints auto-save to Google Drive
4. Free GPU hours (no cost)
5. Can always resume if connection drops
```

### If you have A100 or RTX 3090:
```
1. Can use batch_size=16 or larger
2. Training completes in 2-3 hours
3. Can run multiple epochs in single session
4. Good for parameter tuning
```

---

## 🔍 Monitor Training

### Check GPU Health
```python
# In any cell during training:
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
```

### Watch Logs in Real-Time
```bash
# Local machine:
tail -f cache/logs/pdc_training_*.log

# Colab:
!tail -f cache/logs/pdc_training_*.log
```

### Check Checkpoints
```python
# In notebook:
checkpoints = list(CHECKPOINTS_DIR.glob("*.pt"))
for cp in sorted(checkpoints)[-5:]:
    print(f"{cp.name}: {cp.stat().st_size / 1e6:.1f} MB")
```

---

## 📊 Dataset Info

### OpenWebText (Pretraining - FULL)
- **Size**: ~20GB full dataset
- **First download**: 20GB (multiple hours depending on connection)
- **Tokenization time**: 30-60 min (one-time, then cached)
- **Cached locally**: Subsequent runs skip download but load tokenized data
- **Storage**: cache/datasets/openwebtext/ (20GB+)
- **Split**: 90% train, 10% validation

### GSM8K (Evaluation - FULL)
- **Size**: Full 7,473 training examples
- **First download**: ~500 MB, ~5-10 min
- **Cached locally**: Subsequent runs <1 min
- **Storage**: cache/datasets/gsm8k/
- **Note**: Evaluation runs on all 7,473 examples (~30-60 min)

**Total setup time**:
- First run: 1-2 hours (download + tokenization)
- Subsequent runs: 10-30 min (load cached data, no download)
- Training time: 4-10 hours depending on GPU

---

## 💾 Checkpoint Management

### Automatic Saving
```
Every 100 steps:     → checkpoint_epochX_stepN.pt
After each epoch:    → checkpoint_epochX_final.pt
On interrupt (Ctrl+C): → checkpoint_emergency.pt
On error:            → checkpoint_error.pt
```

### Resume Training
```python
# Automatic in Step 7
# You're asked: "Resume from checkpoint? (y/n):"
# Answer 'y' to continue from last checkpoint

# Manual resume:
latest = checkpoint_manager.get_latest()
checkpoint_manager.load(model, optimizer, latest)
```

---

## 🆘 Support Checklist

- [ ] GPU available: `torch.cuda.is_available()`?
- [ ] Enough memory: `torch.cuda.memory_allocated() < 0.8 * total`?
- [ ] Dataset cached: `(CACHE_DIR / 'datasets').exists()`?
- [ ] Checkpoints saving: `(CHECKPOINTS_DIR).exists()`?
- [ ] Logs being written: `(LOGS_DIR).exists()`?

If any false, check that cell for errors and re-run.

---

## Quick Reference: Batch Size by GPU

```
GPU Memory    Safe Batch Size    Recommended Config
────────────────────────────────────────────────────
4GB           2-4                batch=2, hidden=64, layers=2
8GB (4050)    4-8                batch=4, hidden=256, layers=4
12GB          8-16               batch=8, hidden=256, layers=4
16GB (T4)     8-16               batch=8, hidden=256, layers=4
24GB+         16-32              batch=16, hidden=512, layers=6+
```

---

**Last Updated**: May 2026  
**Tested On**: Colab T4, Local RTX 4050, RTX 3090
