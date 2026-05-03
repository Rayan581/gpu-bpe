"""
Hybrid Semantic Guard (HSG): digit-span locking for numeric fidelity.

Preserves digit sequences as contiguous tokens to prevent BlockBPE
accuracy drop on mathematical tasks (GSM8K, MATH).
"""

from typing import List, Tuple, Optional
import re


class DigitSpanLocker:
    """
    Pre-processing pass that locks digit spans (0x30-0x39) before tokenization.
    Post-processing pass that unlocks them after tokenization.

    Ensures digits remain contiguous tokens, preventing loss of numeric fidelity
    due to byte-level BPE merges across digit boundaries.
    """

    # Special tokens for digit span wrapping
    DIGIT_START = "<|digit_start|>"
    DIGIT_END = "<|digit_end|>"
    DIGIT_SPAN_PATTERN = re.compile(r'\d+')

    def __init__(self, enable: bool = True):
        self.enable = enable
        self.digit_spans = []  # Track original digit spans for reconstruction

    def lock_digit_spans(self, text: str) -> str:
        """
        Wrap digit sequences with special markers to prevent BPE merging.

        Args:
            text: Input text

        Returns:
            Text with digit spans wrapped in special tokens
        """
        if not self.enable:
            return text

        self.digit_spans = []
        result = []
        last_end = 0

        for match in self.DIGIT_SPAN_PATTERN.finditer(text):
            # Add text before digit span
            result.append(text[last_end:match.start()])

            # Wrap digit span
            digit_span = match.group()
            self.digit_spans.append((len(''.join(result)), digit_span))
            result.append(self.DIGIT_START)
            result.append(digit_span)
            result.append(self.DIGIT_END)

            last_end = match.end()

        result.append(text[last_end:])
        return ''.join(result)

    def unlock_digit_spans(self, text: str) -> str:
        """
        Restore original digit spans (inverse of lock_digit_spans).

        Args:
            text: Text with wrapped digit spans

        Returns:
            Text with unwrapped digit spans
        """
        if not self.enable:
            return text

        # Simple replacement for wrapped markers
        text = text.replace(self.DIGIT_START, '')
        text = text.replace(self.DIGIT_END, '')
        return text

    def lock_token_ids(
        self,
        token_ids: List[int],
        tokenizer,
        text: str
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Lock digit spans at token level (alternative approach).

        Returns token IDs and list of (start_idx, end_idx) for digit spans.
        """
        if not self.enable:
            return token_ids, []

        # Find digit spans in original text
        digit_spans_info = []
        for match in self.DIGIT_SPAN_PATTERN.finditer(text):
            digit_spans_info.append((match.start(), match.end(), match.group()))

        # Map character positions to token positions (simplified)
        # In practice, use alignment from tokenizer
        locked_spans = []
        for char_start, char_end, digit_str in digit_spans_info:
            # Approximate token indices
            token_start = max(0, char_start // 4)  # Rough estimate
            token_end = min(len(token_ids), char_end // 4 + 1)
            locked_spans.append((token_start, token_end))

        return token_ids, locked_spans


class SemanticGuardedTokenizer:
    """
    Tokenizer wrapper that applies Hybrid Semantic Guard before encoding.
    """

    def __init__(self, base_tokenizer, enable_hsg: bool = True):
        self.base_tokenizer = base_tokenizer
        self.hsg = DigitSpanLocker(enable=enable_hsg)
        self.enable_hsg = enable_hsg

    def encode(self, text: str):
        """Encode with HSG digit-span locking."""
        if self.enable_hsg:
            locked_text = self.hsg.lock_digit_spans(text)
            token_ids = self.base_tokenizer.encode(locked_text)
        else:
            token_ids = self.base_tokenizer.encode(text)

        return token_ids

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None):
        """Encode batch with HSG."""
        if self.enable_hsg:
            locked_texts = [self.hsg.lock_digit_spans(text) for text in texts]
            return self.base_tokenizer.encode_batch(
                locked_texts,
                max_length=max_length
            )
        else:
            return self.base_tokenizer.encode_batch(texts, max_length=max_length)

    def decode(self, token_ids):
        """Decode and restore digit spans."""
        text = self.base_tokenizer.decode(token_ids)
        if self.enable_hsg:
            text = self.hsg.unlock_digit_spans(text)
        return text

    def decode_batch(self, token_ids_batch):
        """Decode batch and restore digit spans."""
        texts = self.base_tokenizer.decode_batch(token_ids_batch)
        if self.enable_hsg:
            texts = [self.hsg.unlock_digit_spans(text) for text in texts]
        return texts


def test_digit_span_locking():
    """Simple test for digit-span locking."""
    locker = DigitSpanLocker(enable=True)

    test_cases = [
        "The answer is 42.",
        "2 + 2 = 4",
        "GSM8K: 123 apples and 456 oranges",
        "Pi is 3.14159",
        "No digits here",
    ]

    for text in test_cases:
        locked = locker.lock_digit_spans(text)
        unlocked = locker.unlock_digit_spans(locked)
        print(f"Original:  {text}")
        print(f"Locked:    {locked}")
        print(f"Unlocked:  {unlocked}")
        print(f"Match: {text == unlocked}\n")


if __name__ == "__main__":
    test_digit_span_locking()
