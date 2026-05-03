"""
Data loading utilities for training and evaluation.

Provides:
- OpenWebText subset for pretraining
- GSM8K for math evaluation
- MATH subset for math evaluation
- Synthetic math problems
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import random
import os
import json
from pathlib import Path


class TextDataset(Dataset):
    """Generic text dataset for language modeling."""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 512
    ):
        """
        Args:
            texts: list of text documents
            tokenizer: tokenizer to use
            max_length: max sequence length
            stride: stride for sliding window (for multiple sequences from long docs)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = self._create_examples()

    def _create_examples(self) -> List[Dict]:
        """Create training examples from texts."""
        examples = []

        for text in self.texts:
            # Tokenize
            token_ids, _ = self.tokenizer.encode([text])
            if not token_ids:
                continue

            token_ids = token_ids[0]

            # Create sliding window examples
            for i in range(0, len(token_ids) - self.max_length, self.stride):
                input_ids = token_ids[i:i + self.max_length]
                labels = token_ids[i + 1:i + self.max_length + 1]

                if len(input_ids) == self.max_length and len(labels) == self.max_length:
                    examples.append({
                        'input_ids': input_ids,
                        'labels': labels
                    })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


class GSM8KDataset(Dataset):
    """GSM8K math word problem dataset."""

    def __init__(
        self,
        problems: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            problems: list of {question, answer, answer_type}
            tokenizer: tokenizer
            max_length: max sequence length
        """
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        problem = self.problems[idx]

        # Combine question and answer for training
        full_text = f"{problem['question']} {problem['answer']}"

        token_ids, _ = self.tokenizer.encode([full_text])
        if not token_ids or not token_ids[0]:
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long),
                'answer_type': problem.get('answer_type', 'unknown')
            }

        token_ids = token_ids[0]

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [50256] * \
                (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'answer_type': problem.get('answer_type', 'unknown')
        }


class DataLoader_:
    """Simple data loader wrapper (fallback if torch DataLoader unavailable)."""

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = {
                'input_ids': [],
                'labels': []
            }

            for idx in batch_indices:
                item = self.dataset[idx]
                batch['input_ids'].append(item['input_ids'])
                batch['labels'].append(item['labels'])

            # Stack into tensors
            batch['input_ids'] = torch.stack(batch['input_ids'])
            batch['labels'] = torch.stack(batch['labels'])

            yield batch


def get_cache_dir(dataset_type: str, cache_root: str = "cache/datasets") -> Path:
    """Get cache directory for dataset."""
    cache_path = Path(cache_root) / dataset_type
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def save_dataset_cache(data: List, dataset_type: str, cache_root: str = "cache/datasets"):
    """Save dataset to cache."""
    cache_dir = get_cache_dir(dataset_type, cache_root)
    cache_file = cache_dir / "data.json"

    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Cached {dataset_type} dataset to {cache_file}")


def load_dataset_cache(dataset_type: str, cache_root: str = "cache/datasets") -> Optional[List]:
    """Load dataset from cache if available."""
    cache_dir = get_cache_dir(dataset_type, cache_root)
    cache_file = cache_dir / "data.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {dataset_type} dataset from cache ({len(data)} items)")
        return data

    return None


def create_openwebtext_subset(
    num_docs: int = 1000,
    max_doc_length: int = 2000,
    cache_root: str = "cache/datasets"
) -> List[str]:
    """
    Create synthetic OpenWebText-like dataset.

    Checks cache first; if found, loads from cache. Otherwise generates new data.

    In practice, would use datasets.load_dataset('openwebtext')
    """
    # Check cache first
    cached = load_dataset_cache('openwebtext', cache_root)
    if cached is not None:
        return cached
    topics = [
        "machine learning", "deep learning", "neural networks",
        "transformers", "language models", "training",
        "optimization", "distributed systems", "GPU computing",
        "quantization", "compression", "tokenization",
        "natural language processing", "computer vision",
        "reinforcement learning", "generative models"
    ]

    docs = []
    for _ in range(num_docs):
        # Generate synthetic document
        topic = random.choice(topics)
        length = random.randint(500, max_doc_length)

        text = f"About {topic}: "
        text += " ".join(random.choice(topics) for _ in range(length // 15))

        docs.append(text)

    # Save to cache
    save_dataset_cache(docs, 'openwebtext', cache_root)
    return docs


def create_gsm8k_subset(num_problems: int = 100, cache_root: str = "cache/datasets") -> List[Dict]:
    """
    Create synthetic GSM8K-like dataset.

    Checks cache first; if found, loads from cache. Otherwise generates new data.

    In practice, would use datasets.load_dataset('gsm8k', 'main')
    """
    # Check cache first
    cached = load_dataset_cache('gsm8k', cache_root)
    if cached is not None:
        return cached
    problems = []

    for i in range(num_problems):
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)
        num3 = random.randint(1, 50)

        if i % 3 == 0:
            # Addition/subtraction
            question = f"If there are {num1} apples and {num2} oranges, how many fruits are there?"
            answer = f"The answer is {num1 + num2}."
        elif i % 3 == 1:
            # Multiplication
            question = f"If each of {num1} people has {num2} dollars, how much in total?"
            answer = f"The answer is {num1 * num2} dollars."
        else:
            # Division
            question = f"If {num1} candies are shared among {num2} children, how many each?"
            answer = f"The answer is {num1 // num2} candies each."

        problems.append({
            'question': question,
            'answer': answer,
            'answer_type': 'arithmetic'
        })

    # Save to cache
    save_dataset_cache(problems, 'gsm8k', cache_root)
    return problems


def create_math_subset(num_problems: int = 100, cache_root: str = "cache/datasets") -> List[Dict]:
    """Create synthetic MATH dataset.

    Checks cache first; if found, loads from cache. Otherwise generates new data.
    """
    # Check cache first
    cached = load_dataset_cache('math', cache_root)
    if cached is not None:
        return cached
    problems = []

    for i in range(num_problems):
        a = random.randint(1, 10)
        b = random.randint(1, 10)

        if i % 4 == 0:
            question = f"What is {a}x + {b} when x = 5?"
            answer = f"The answer is {a * 5 + b}."
        elif i % 4 == 1:
            question = f"Solve: {a}x + {b} = 100"
            answer = f"x = {(100 - b) / a:.1f}"
        elif i % 4 == 2:
            question = f"What is the area of rectangle with sides {a} and {b}?"
            answer = f"The area is {a * b} square units."
        else:
            question = f"What is {a}^2 + {b}^2?"
            answer = f"The answer is {a**2 + b**2}."

        problems.append({
            'question': question,
            'answer': answer,
            'answer_type': 'algebra'
        })

    # Save to cache
    save_dataset_cache(problems, 'math', cache_root)
    return problems


def get_dataloader(
    dataset_type: str = 'owt',
    tokenizer=None,
    batch_size: int = 32,
    num_docs: int = 100,
    max_length: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
    cache_root: str = "cache/datasets"
):
    """
    Create dataloader for specified dataset.

    Args:
        dataset_type: 'owt', 'gsm8k', 'math', or 'synthetic'
        tokenizer: tokenizer instance
        batch_size: batch size
        num_docs: number of documents/problems
        max_length: max sequence length
        num_workers: number of workers for dataloader
        pin_memory: pin memory for GPU
        cache_root: root directory for caching datasets

    Returns:
        DataLoader instance
    """
    if dataset_type == 'owt':
        texts = create_openwebtext_subset(
            num_docs=num_docs, cache_root=cache_root)
        dataset = TextDataset(texts, tokenizer, max_length=max_length)
    elif dataset_type == 'gsm8k':
        problems = create_gsm8k_subset(
            num_problems=num_docs, cache_root=cache_root)
        dataset = GSM8KDataset(problems, tokenizer, max_length=max_length)
    elif dataset_type == 'math':
        problems = create_math_subset(
            num_problems=num_docs, cache_root=cache_root)
        dataset = GSM8KDataset(problems, tokenizer, max_length=max_length)
    elif dataset_type == 'synthetic':
        texts = [f"Synthetic text {i} " * 50 for i in range(num_docs)]
        dataset = TextDataset(texts, tokenizer, max_length=max_length)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    try:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    except:
        # Fallback to simple loader
        return DataLoader_(dataset, batch_size=batch_size, shuffle=True)


def test_dataloader():
    """Test data loading."""
    from tokenizer.gpu_bpe import GPUBPETokenizer

    tokenizer = GPUBPETokenizer(vocab_size=50257)

    # Test text dataset
    loader = get_dataloader('owt', tokenizer, batch_size=4, num_docs=10)
    print("OpenWebText loader created")

    batch = next(iter(loader))
    print(f"Batch shape: {batch['input_ids'].shape}\n")

    # Test GSM8K dataset
    loader = get_dataloader('gsm8k', tokenizer, batch_size=4, num_docs=10)
    print("GSM8K loader created")

    batch = next(iter(loader))
    print(f"GSM8K batch shape: {batch['input_ids'].shape}\n")


if __name__ == "__main__":
    test_dataloader()
