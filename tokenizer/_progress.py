"""
Shared progress bar helper for tokenizer training.

Uses tqdm.notebook in Colab (renders as a widget, visible in cell output),
falls back to tqdm.auto elsewhere.
"""

def get_tqdm():
    """Return the right tqdm class for the current environment."""
    try:
        # Check if we're inside a Jupyter/Colab kernel
        get_ipython()  # noqa: F821 — defined in kernel globals
        try:
            from tqdm.notebook import tqdm
            return tqdm
        except ImportError:
            pass
    except NameError:
        pass  # not in a notebook

    try:
        from tqdm.auto import tqdm
        return tqdm
    except ImportError:
        return None