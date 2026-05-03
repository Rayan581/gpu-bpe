#!/usr/bin/env python3
"""
Main entry point for running experiments.

Usage:
    python run_experiment.py --mode baseline --steps 100
    python run_experiment.py --mode full --steps 200 --compression
    python run_experiment.py --mode ablation-a --steps 100
    python run_experiment.py --mode ablation-b --steps 100
    python run_experiment.py --eval-gsm8k --checkpoint outputs/full/latest.pt
"""

import argparse
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_baseline(args):
    """Run baseline training (CPU tokenizer, sync FP32)."""
    import subprocess

    cmd = [
        "python", "scripts/train_baseline.py",
        "--num_steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--max_length", str(args.max_length),
        "--learning_rate", str(args.learning_rate),
        "--save_dir", args.save_dir,
    ]

    print(f"Running baseline training...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_full(args):
    """Run full system training."""
    import subprocess

    cmd = [
        "python", "scripts/train_full.py",
        "--num_steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--max_length", str(args.max_length),
        "--learning_rate", str(args.learning_rate),
        "--num_workers", str(args.num_workers),
        "--save_dir", args.save_dir,
    ]

    if args.compression:
        cmd.append("--compression_enabled")
    if args.hsg:
        cmd.append("--hsg_enabled")

    print(f"Running full system training...")
    print(f"  GPU tokenizer: ✓")
    print(f"  HSG (digit-span locking): {'✓' if args.hsg else '✗'}")
    print(f"  Compression (INT8/INT4): {'✓' if args.compression else '✗'}")
    print(f"  DC-ASGD: ✓")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_ablation_a(args):
    """Ablation A: GPU + three-tier, NO HSG."""
    import subprocess

    cmd = [
        "python", "scripts/ablation_a.py",
        "--num_steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--max_length", str(args.max_length),
        "--save_dir", args.save_dir,
    ]

    print(f"Running ablation A (GPU, NO HSG)...")
    print(f"  GPU tokenizer: ✓")
    print(f"  HSG: ✗")
    print(f"  Compression: ✓")
    print(f"Compare with Full System to measure HSG impact\n")
    subprocess.run(cmd)


def run_ablation_b(args):
    """Ablation B: GPU + HSG, NO compression."""
    import subprocess

    cmd = [
        "python", "scripts/ablation_b.py",
        "--num_steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--max_length", str(args.max_length),
        "--save_dir", args.save_dir,
    ]

    print(f"Running ablation B (GPU + HSG, NO compression)...")
    print(f"  GPU tokenizer: ✓")
    print(f"  HSG: ✓")
    print(f"  Compression: ✗")
    print(f"Compare with Full System to measure compression impact\n")
    subprocess.run(cmd)


def eval_gsm8k(args):
    """Evaluate on GSM8K dataset."""
    import subprocess

    cmd = [
        "python", "scripts/eval_gsm8k.py",
        "--checkpoint", args.checkpoint,
        "--num_problems", str(args.num_problems),
    ]

    if args.save_dir:
        cmd.extend(["--save_dir", args.save_dir])

    print(f"Evaluating on GSM8K ({args.num_problems} problems)...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def eval_perplexity(args):
    """Evaluate perplexity."""
    import subprocess

    cmd = [
        "python", "scripts/eval_perplexity.py",
        "--checkpoint", args.checkpoint,
        "--split", args.split,
    ]

    if args.save_dir:
        cmd.extend(["--save_dir", args.save_dir])

    print(f"Evaluating perplexity on {args.split} set...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def list_outputs():
    """List recent outputs."""
    outputs_dir = PROJECT_ROOT / "outputs"
    cache_dir = PROJECT_ROOT / "cache"

    print("\n" + "="*60)
    print("RECENT OUTPUTS")
    print("="*60 + "\n")

    # List outputs/ directory
    if outputs_dir.exists():
        print("Script outputs (outputs/):")
        for subdir in sorted(outputs_dir.iterdir()):
            if subdir.is_dir():
                metrics_file = subdir / "metrics.json"
                if metrics_file.exists():
                    print(f"  ✓ {subdir.name}/")
                    print(f"    - metrics.json ({metrics_file.stat().st_size / 1e3:.0f} KB)")

    # List cache/ directory
    if cache_dir.exists():
        logs_dir = cache_dir / "logs"
        checkpoints_dir = cache_dir / "checkpoints"

        if logs_dir.exists():
            logs = list(logs_dir.glob("*.log"))
            if logs:
                print(f"\n  Jupyter Notebook logs (cache/logs/):")
                for log in sorted(logs)[-5:]:
                    print(f"    - {log.name}")

        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.pt"))
            if checkpoints:
                print(f"\n  Checkpoints (cache/checkpoints/):")
                for cp in sorted(checkpoints)[-5:]:
                    size_mb = cp.stat().st_size / 1e6
                    print(f"    - {cp.name} ({size_mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="PDC Project: GPU-Accelerated Parallel BPE Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline training
  python run_experiment.py --mode baseline --steps 100

  # Run full system with compression and HSG
  python run_experiment.py --mode full --steps 200 --compression --hsg

  # Run ablation studies
  python run_experiment.py --mode ablation-a --steps 100
  python run_experiment.py --mode ablation-b --steps 100

  # Evaluate trained model
  python run_experiment.py --eval-gsm8k --checkpoint outputs/full/latest.pt

  # List all outputs
  python run_experiment.py --list-outputs
        """
    )

    # Experiment mode
    parser.add_argument(
        "--mode",
        choices=["baseline", "full", "ablation-a", "ablation-b"],
        help="Training mode (choose baseline vs full vs ablations)"
    )

    # Training parameters
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max sequence length (default: 256)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of workers for full system (default: 2)")

    # Features (only for full mode)
    parser.add_argument("--compression", action="store_true",
                        help="Enable compression (INT8/INT4 quantization)")
    parser.add_argument("--hsg", action="store_true",
                        help="Enable Hybrid Semantic Guard (digit-span locking)")

    # Evaluation
    parser.add_argument("--eval-gsm8k", action="store_true",
                        help="Evaluate on GSM8K dataset")
    parser.add_argument("--eval-perplexity", action="store_true",
                        help="Evaluate perplexity")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint for evaluation")
    parser.add_argument("--num-problems", type=int, default=100,
                        help="Number of GSM8K problems to evaluate (default: 100)")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation", "test"],
                        help="Data split for evaluation (default: validation)")

    # Output
    parser.add_argument("--save-dir", type=str, default="./outputs/experiment",
                        help="Directory to save results (default: ./outputs/experiment)")
    parser.add_argument("--list-outputs", action="store_true",
                        help="List recent outputs and exit")

    args = parser.parse_args()

    # Handle list outputs
    if args.list_outputs:
        list_outputs()
        return

    # Handle evaluation without mode
    if args.eval_gsm8k:
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation")
            sys.exit(1)
        eval_gsm8k(args)
        return

    if args.eval_perplexity:
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation")
            sys.exit(1)
        eval_perplexity(args)
        return

    # Handle training modes
    if not args.mode:
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK GUIDE")
        print("="*60)
        print("""
For beginners:
  1. Test setup: python tokenizer/gpu_bpe.py
  2. Quick baseline: python run_experiment.py --mode baseline --steps 20
  3. Full system: python run_experiment.py --mode full --steps 20

For full training on actual data:
  → Use Jupyter notebook (train_notebook.ipynb) - recommended

For reproducible scripts:
  python run_experiment.py --mode full --steps 500 --compression --hsg

For ablation studies:
  python run_experiment.py --mode ablation-a --steps 500
  python run_experiment.py --mode ablation-b --steps 500

For evaluation:
  python run_experiment.py --eval-gsm8k --checkpoint outputs/full/latest.pt
        """)
        sys.exit(0)

    # Run selected mode
    print("="*60)
    print(f"PDC Project: GPU-Accelerated Parallel BPE Training")
    print("="*60 + "\n")

    if args.mode == "baseline":
        run_baseline(args)
    elif args.mode == "full":
        run_full(args)
    elif args.mode == "ablation-a":
        run_ablation_a(args)
    elif args.mode == "ablation-b":
        run_ablation_b(args)

    # Show where results were saved
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.save_dir}")
    print("\nNext steps:")
    print(f"  - Review metrics: cat {args.save_dir}/metrics.json")
    print(f"  - Evaluate: python run_experiment.py --eval-gsm8k --checkpoint {args.save_dir}/latest.pt")
    print(f"  - List outputs: python run_experiment.py --list-outputs")


if __name__ == "__main__":
    main()
