"""
Evaluate model perplexity on a held-out test set.

Measures language modeling quality without regression from tokenizer changes.

Usage:
    python eval_perplexity.py --checkpoint ./model.pt --num_docs 100
"""

from torch.utils.data import DataLoader
from utils.data import load_wikitext, TextDataset
from tokenizer.hsg import SemanticGuardedTokenizer
from tokenizer.gpu_bpe import GPUBPETokenizer
import torch
import torch.nn as nn
import sys
import argparse
from pathlib import Path
import json
import math

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def evaluate_perplexity(
    checkpoint: str = None,
    num_docs: int = 100,
    batch_size: int = 16,
    max_length: int = 512,
    use_hsg: bool = True,
    save_dir: str = "./outputs/eval",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate perplexity on held-out test set.

    Args:
        checkpoint: path to saved model
        num_docs: number of documents for eval
        batch_size: batch size
        max_length: max sequence length
        use_hsg: whether to use HSG
        save_dir: output directory
        device: compute device
    """
    print(f"=== Perplexity Evaluation ===")
    print(f"Num docs: {num_docs}")
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
    print(f"Creating evaluation dataset with {num_docs} documents...")
    texts = load_wikitext(num_docs=num_docs)
    dataset = TextDataset(texts, tokenizer, max_length=max_length)

    try:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except:
        # Fallback
        from utils.data import DataLoader_
        dataloader = DataLoader_(dataset, batch_size=batch_size, shuffle=False)

    # Evaluation loop
    print("\nEvaluating perplexity...")
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            batch_loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Count valid tokens (not padding)
            valid_mask = (labels != 50256).float()
            num_valid = valid_mask.sum().item()

            total_loss += batch_loss.item()
            total_tokens += num_valid
            num_batches += 1

            if num_batches % max(1, (len(dataset) // batch_size) // 10) == 0:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                ppl = math.exp(avg_loss) if avg_loss > 0 else float('inf')
                print(
                    f"Batch {num_batches}: "
                    f"loss={avg_loss:.4f}, perplexity={ppl:.2f}"
                )

    # Final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')

    print(f"\n=== Perplexity Results ===")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Evaluated on {total_tokens:.0f} tokens\n")

    # Save results
    output_file = f"{save_dir}/perplexity_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'perplexity': perplexity,
            'loss': avg_loss,
            'total_tokens': int(total_tokens),
            'num_batches': num_batches,
            'use_hsg': use_hsg
        }, f, indent=2)

    print(f"Results saved to {output_file}\n")

    return perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_hsg", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./outputs/eval")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_perplexity(
        checkpoint=args.checkpoint,
        num_docs=args.num_docs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_hsg=args.use_hsg,
        save_dir=args.save_dir,
        device=device
    )
