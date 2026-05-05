"""
Data loading utilities for training and evaluation.

Provides:
- WikiText-103 (document-level) for pretraining corpus
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
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = self._create_examples()

    def _create_examples(self) -> List[Dict]:
        examples = []

        for text in self.texts:
            token_ids, _ = self.tokenizer.encode([text])
            if not token_ids:
                continue

            token_ids = token_ids[0]

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
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        problem = self.problems[idx]

        full_text = f"{problem['question']} {problem['answer']}"

        token_ids, _ = self.tokenizer.encode([full_text])
        if not token_ids or not token_ids[0]:
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long),
                'answer_type': problem.get('answer_type', 'unknown')
            }

        token_ids = token_ids[0]

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
            batch = {'input_ids': [], 'labels': []}

            for idx in batch_indices:
                item = self.dataset[idx]
                batch['input_ids'].append(item['input_ids'])
                batch['labels'].append(item['labels'])

            batch['input_ids'] = torch.stack(batch['input_ids'])
            batch['labels'] = torch.stack(batch['labels'])

            yield batch


def get_cache_dir(dataset_type: str, cache_root: str = "cache/datasets") -> Path:
    cache_path = Path(cache_root) / dataset_type
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def save_dataset_cache(data: List, dataset_type: str, cache_root: str = "cache/datasets"):
    cache_dir = get_cache_dir(dataset_type, cache_root)
    cache_file = cache_dir / "data.json"

    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Cached {dataset_type} dataset to {cache_file}")


def load_dataset_cache(dataset_type: str, cache_root: str = "cache/datasets") -> Optional[List]:
    cache_dir = get_cache_dir(dataset_type, cache_root)
    cache_file = cache_dir / "data.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {dataset_type} dataset from cache ({len(data)} items)")
        return data

    return None


def load_wikitext(
    num_docs: int = 1000,
    min_doc_length: int = 100,
    cache_root: str = "cache/datasets"
) -> List[str]:
    """
    Load WikiText-103 (document-level) from HuggingFace.

    Uses EleutherAI/wikitext_document_level with the wikitext-103-raw-v1
    subset. Each row is a full Wikipedia article — no near-empty header
    lines polluting the corpus, unlike the line-by-line variant.

    Falls back to cache on subsequent runs to avoid repeated downloads.

    Args:
        num_docs:        number of documents to load from the train split
        min_doc_length:  minimum character length to filter empty/stub articles
        cache_root:      root directory for local caching

    Returns:
        List of document strings
    """
    cache_key = f"wikitext_{num_docs}"
    cached = load_dataset_cache(cache_key, cache_root)
    if cached is not None:
        return cached

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace 'datasets' package is required. "
            "Install with: pip install datasets"
        )

    print(f"Loading WikiText-103 ({num_docs} docs)...")

    # Try document-level variant first (one row = one full article).
    # Fall back to the line-by-line Salesforce/wikitext and group by article
    # boundary (lines starting with ' = ' are article headings).
    docs = []
    try:
        dataset = load_dataset(
            "EleutherAI/wikitext_document_level",
            "wikitext-103-raw-v1",
            split="train",
        )
        for row in dataset:
            text = row["text"].strip()
            if len(text) >= min_doc_length:
                docs.append(text)
            if len(docs) >= num_docs:
                break
        print(f"  Collected {len(docs)} docs via document-level variant")
    except Exception as e:
        print(f"  Document-level variant unavailable ({e}), "
              f"falling back to Salesforce/wikitext ...")
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
        )
        # Group consecutive lines into articles using heading markers
        current: list = []
        for row in dataset:
            line = row["text"]
            if line.startswith(" = ") and not line.startswith(" = = ") and current:
                doc = " ".join(current).strip()
                if len(doc) >= min_doc_length:
                    docs.append(doc)
                if len(docs) >= num_docs:
                    break
                current = [line]
            else:
                current.append(line)
        # flush last article
        if current and len(docs) < num_docs:
            doc = " ".join(current).strip()
            if len(doc) >= min_doc_length:
                docs.append(doc)
        print(f"  Collected {len(docs)} docs via Salesforce/wikitext fallback")

    save_dataset_cache(docs, cache_key, cache_root)
    return docs


def create_gsm8k_subset(
    num_problems: int = 100,
    cache_root: str = "cache/datasets"
) -> List[Dict]:
    """
    Load real GSM8K problems from HuggingFace, with local cache.

    Falls back to a synthetic generator if the download fails
    (e.g., no internet connection in a restricted lab environment).
    """
    cache_key = f"gsm8k_{num_problems}"
    cached = load_dataset_cache(cache_key, cache_root)
    if cached is not None:
        return cached

    try:
        from datasets import load_dataset as hf_load
        print(f"Loading GSM8K ({num_problems} problems)...")
        hf_ds = hf_load("gsm8k", "main", split="train")
        problems = []
        for row in hf_ds:
            problems.append({
                "question": row["question"],
                "answer": row["answer"],
                "answer_type": "arithmetic",
            })
            if len(problems) >= num_problems:
                break
        print(f"  Loaded {len(problems)} GSM8K problems from HuggingFace")
    except Exception as e:
        print(f"  HuggingFace download failed ({e}), using synthetic fallback")
        problems = _synthetic_gsm8k(num_problems)

    save_dataset_cache(problems, cache_key, cache_root)
    return problems


def _synthetic_gsm8k(num_problems: int) -> List[Dict]:
    """Synthetic GSM8K-style problems (offline fallback)."""
    problems = []
    for i in range(num_problems):
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)

        if i % 3 == 0:
            question = (f"If there are {num1} apples and {num2} oranges, "
                        f"how many fruits are there?")
            answer = f"The answer is {num1 + num2}."
        elif i % 3 == 1:
            question = (f"If each of {num1} people has {num2} dollars, "
                        f"how much in total?")
            answer = f"The answer is {num1 * num2} dollars."
        else:
            question = (f"If {num1} candies are shared among {num2} children, "
                        f"how many each?")
            answer = f"The answer is {num1 // max(num2, 1)} candies each."

        problems.append({
            "question": question,
            "answer": answer,
            "answer_type": "arithmetic",
        })
    return problems


def create_math_subset(
    num_problems: int = 100,
    cache_root: str = "cache/datasets"
) -> List[Dict]:
    """Synthetic algebra problems (lightweight offline dataset)."""
    cache_key = f"math_{num_problems}"
    cached = load_dataset_cache(cache_key, cache_root)
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
            question = f"What is the area of a rectangle with sides {a} and {b}?"
            answer = f"The area is {a * b} square units."
        else:
            question = f"What is {a}^2 + {b}^2?"
            answer = f"The answer is {a**2 + b**2}."

        problems.append({
            "question": question,
            "answer": answer,
            "answer_type": "algebra",
        })

    save_dataset_cache(problems, cache_key, cache_root)
    return problems


def get_dataloader(
    dataset_type: str = 'wikitext',
    tokenizer=None,
    batch_size: int = 32,
    num_docs: int = 100,
    max_length: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
    cache_root: str = "cache/datasets"
):
    """
    Create a DataLoader for the specified dataset.

    Args:
        dataset_type: 'wikitext', 'gsm8k', 'math', or 'synthetic'
        tokenizer:    tokenizer instance
        batch_size:   batch size
        num_docs:     number of documents / problems to load
        max_length:   max token sequence length
        num_workers:  DataLoader worker processes
        pin_memory:   pin memory for faster GPU transfers
        cache_root:   root directory for dataset cache

    Returns:
        DataLoader instance
    """
    if dataset_type == 'wikitext':
        texts = load_wikitext(num_docs=num_docs, cache_root=cache_root)
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
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Choose from: 'wikitext', 'gsm8k', 'math', 'synthetic'"
        )

    try:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    except Exception:
        return DataLoader_(dataset, batch_size=batch_size, shuffle=True)


def test_dataloader():
    """Quick smoke-test for data loading."""
    from tokenizer.gpu_bpe import GPUBPETokenizer

    tokenizer = GPUBPETokenizer(vocab_size=50257)

    loader = get_dataloader('wikitext', tokenizer, batch_size=4, num_docs=10)
    print("WikiText-103 loader created")
    batch = next(iter(loader))
    print(f"Batch shape: {batch['input_ids'].shape}\n")

    loader = get_dataloader('gsm8k', tokenizer, batch_size=4, num_docs=10)
    print("GSM8K loader created")
    batch = next(iter(loader))
    print(f"GSM8K batch shape: {batch['input_ids'].shape}\n")
