"""
Evaluate model on GSM8K (math word problems).

Tests if byte-level tokenization preserves math accuracy.

Usage:
    python eval_gsm8k.py --checkpoint ./model.pt --num_problems 100
"""

import torch
import torch.nn as nn
import sys
import argparse
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizer.gpu_bpe import GPUBPETokenizer
from tokenizer.hsg import SemanticGuardedTokenizer
from utils.data import create_gsm8k_subset, GSM8KDataset
from utils.metrics import EvaluationMetrics


def create_model(vocab_size: int = 50257, hidden_size: int = 768, num_layers: int = 12):
    """Create GPT-2 Small model."""
    return nn.Sequential(
        nn.Embedding(vocab_size, hidden_size),
        *[
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(num_layers)
        ],
        nn.Linear(hidden_size, vocab_size)
    )


def extract_answer(text: str):
    """Simple answer extraction from model output."""
    # Look for patterns like "The answer is X"
    lines = text.split('\n')
    for line in reversed(lines):
        if 'answer' in line.lower():
            # Try to extract number
            tokens = line.split()
            for token in reversed(tokens):
                try:
                    return int(token.replace('.', '').replace(',', ''))
                except:
                    pass
    return None


def evaluate_gsm8k(
    checkpoint: str = None,
    num_problems: int = 100,
    batch_size: int = 16,
    max_length: int = 512,
    use_hsg: bool = True,
    save_dir: str = "./outputs/eval",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate on GSM8K.

    Args:
        checkpoint: path to saved model
        num_problems: number of problems to evaluate
        batch_size: batch size
        max_length: max sequence length
        use_hsg: whether to use HSG
        save_dir: output directory
        device: compute device
    """
    print(f"=== GSM8K Evaluation ===")
    print(f"Num problems: {num_problems}")
    print(f"Use HSG: {use_hsg}")
    print(f"Device: {device}\n")

    device = torch.device(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Setup tokenizer
    print("Setting up tokenizer...")
    base_tokenizer = GPUBPETokenizer(vocab_size=50257, use_gpu=True)
    if use_hsg:
        tokenizer = SemanticGuardedTokenizer(base_tokenizer, enable_hsg=True)
    else:
        tokenizer = base_tokenizer

    # Load model
    print("Loading model...")
    model = create_model()
    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Create dataset
    print(f"Creating GSM8K dataset with {num_problems} problems...")
    problems = create_gsm8k_subset(num_problems=num_problems)
    dataset = GSM8KDataset(problems, tokenizer, max_length=max_length)

    # Evaluation loop
    print("\nEvaluating...")
    correct = 0
    total = 0
    results = []

    for idx in range(min(num_problems, len(dataset))):
        item = dataset[idx]
        input_ids = item['input_ids'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_ids)

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        predicted_text = tokenizer.decode(predictions[0])

        # Extract answers
        predicted_answer = extract_answer(predicted_text)
        problem = problems[idx]
        expected_answer_str = problem['answer']

        # Simple check: see if predicted answer appears in expected
        is_correct = False
        if predicted_answer is not None:
            if str(predicted_answer) in expected_answer_str:
                is_correct = True

        results.append({
            'problem_idx': idx,
            'question': problem['question'],
            'expected': expected_answer_str,
            'predicted': predicted_text[:100],
            'correct': is_correct
        })

        if is_correct:
            correct += 1
        total += 1

        if (idx + 1) % max(1, num_problems // 10) == 0:
            print(f"Evaluated {idx + 1}/{num_problems}: {100 * correct / total:.1f}% accurate")

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n=== GSM8K Results ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}\n")

    # Save results
    output_file = f"{save_dir}/gsm8k_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'use_hsg': use_hsg,
            'results': results
        }, f, indent=2)

    print(f"Results saved to {output_file}\n")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_hsg", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./outputs/eval")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_gsm8k(
        checkpoint=args.checkpoint,
        num_problems=args.num_problems,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_hsg=args.use_hsg,
        save_dir=args.save_dir,
        device=device
    )
