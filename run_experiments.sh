#!/bin/bash
# Run all experiments: baseline, ablations, and evaluations

set -e  # Exit on first error

echo "=== GPU-Accelerated Parallel BPE Tokenization ==="
echo "Running complete experiment pipeline..."
echo ""

# Configuration
NUM_STEPS=${1:-50}  # First argument: number of steps (default 50)
BATCH_SIZE=${2:-16}  # Second argument: batch size (default 16)
DEVICE=${3:-cuda}   # Third argument: device (default cuda)

echo "Configuration:"
echo "  NUM_STEPS=$NUM_STEPS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  DEVICE=$DEVICE"
echo ""

# Create output directory
mkdir -p outputs
mkdir -p outputs/baseline
mkdir -p outputs/ablation_a
mkdir -p outputs/ablation_b
mkdir -p outputs/full
mkdir -p outputs/eval

# Step 1: Baseline
echo "========================================"
echo "Step 1: Training Baseline (CPU tokenizer, sync FP32)"
echo "========================================"
python scripts/train_baseline.py \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --max_length 256 \
    --save_dir ./outputs/baseline \
    --device $DEVICE
echo "Baseline complete. Results in ./outputs/baseline/metrics.json"
echo ""

# Step 2: Ablation A
echo "========================================"
echo "Step 2: Ablation A (GPU tokenizer + three-tier, NO HSG)"
echo "========================================"
python scripts/ablation_a.py \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --max_length 256 \
    --num_workers 2 \
    --save_dir ./outputs/ablation_a \
    --device $DEVICE
echo "Ablation A complete. Results in ./outputs/ablation_a/metrics.json"
echo ""

# Step 3: Ablation B
echo "========================================"
echo "Step 3: Ablation B (GPU tokenizer + HSG, NO compression)"
echo "========================================"
python scripts/ablation_b.py \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --max_length 256 \
    --num_workers 2 \
    --save_dir ./outputs/ablation_b \
    --device $DEVICE
echo "Ablation B complete. Results in ./outputs/ablation_b/metrics.json"
echo ""

# Step 4: Full System
echo "========================================"
echo "Step 4: Full System (GPU + HSG + compression + DC-ASGD)"
echo "========================================"
python scripts/train_full.py \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --max_length 256 \
    --num_workers 2 \
    --max_staleness 5 \
    --enable_hsg True \
    --save_dir ./outputs/full \
    --device $DEVICE
echo "Full system complete. Results in ./outputs/full/metrics.json"
echo ""

# Step 5: Evaluation - GSM8K
echo "========================================"
echo "Step 5: Evaluation - GSM8K"
echo "========================================"
python scripts/eval_gsm8k.py \
    --num_problems 50 \
    --batch_size $BATCH_SIZE \
    --max_length 256 \
    --use_hsg True \
    --save_dir ./outputs/eval \
    --device $DEVICE
echo "GSM8K evaluation complete. Results in ./outputs/eval/gsm8k_results.json"
echo ""

# Step 6: Evaluation - Perplexity
echo "========================================"
echo "Step 6: Evaluation - Perplexity"
echo "========================================"
python scripts/eval_perplexity.py \
    --num_docs 50 \
    --batch_size $BATCH_SIZE \
    --max_length 256 \
    --use_hsg True \
    --save_dir ./outputs/eval \
    --device $DEVICE
echo "Perplexity evaluation complete. Results in ./outputs/eval/perplexity_results.json"
echo ""

# Summary
echo "========================================"
echo "EXPERIMENT PIPELINE COMPLETE"
echo "========================================"
echo ""
echo "Results summary:"
echo "  Baseline:      ./outputs/baseline/metrics.json"
echo "  Ablation A:    ./outputs/ablation_a/metrics.json"
echo "  Ablation B:    ./outputs/ablation_b/metrics.json"
echo "  Full System:   ./outputs/full/metrics.json"
echo "  GSM8K:         ./outputs/eval/gsm8k_results.json"
echo "  Perplexity:    ./outputs/eval/perplexity_results.json"
echo ""
echo "Next steps:"
echo "  1. Compare metrics across experiments"
echo "  2. Analyze ablation impacts on throughput/accuracy"
echo "  3. Check convergence patterns in metrics.json"
echo "  4. Evaluate math accuracy preservation (GSM8K)"
echo "  5. Verify no perplexity regression"
echo ""
