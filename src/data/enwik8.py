from collections import Counter
import os
import re
import unicodedata
import torch
from torch.utils.data import DataLoader

from data.util import download_unzip, encode_text, split_dataset
from typing import Tuple, Dict, Optional


def load_enwik8(path: str, verbose: bool = False) -> str:
    """
    Download and unzip the enwik8 dataset if it doesn't exist.

    Args:
        path: The directory where the dataset will be downloaded and unzipped.
        verbose: Whether to print a summary of the loaded text.

    Return:
        Content of the enwik8 dataset.
    """

    url = "http://mattmahoney.net/dc/enwik8.zip"
    zip_name = "enwik8"
    text_path = os.path.join(path, zip_name)
    download_unzip(url, path, zip_name)
    return preprocess_enwik8(path, text_path, verbose)


def get_enwik8_unique_chars(text: str) -> str:
    """
    Get the unique characters in the enwik8 dataset.

    Args:
        text: The enwik8 text.

    Return:
        The unique characters in the enwik8 dataset.
    """

    return sorted(set(text))


def preprocess_enwik8(data_dir: str, text_path: str, verbose: bool = False) -> str:
    """
    Preprocess the enwik8 text.
    1. Remove HTML tags.
    2. Normalize to Unicode.
    3. Remove control characters.
    4. Replace multiple spaces with a single space.

    Args:
        data_dir: The directory where the dataset resides.
        text_path: The path to the enwik8 text.
        verbose: Whether to print a summary of the preprocessed text.

    Return:
        The preprocessed enwik8 text.
    """

    preprocessed_file = "preprocessed_enwik8.txt"
    preprocessed_path = os.path.join(data_dir, preprocessed_file)

    if os.path.exists(preprocessed_path):
        with open(preprocessed_path, "r") as file:
            text = file.read()
    else:
        # Read the text file
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read()
        if verbose:
            print(f"Length of enwik8 dataset: {len(text)}")
            print(f"Sample of enwik8 dataset: {text[:1000]}")
            freq = sorted(Counter(text).items(), key=lambda item: item[1], reverse=True)
            print(f"Frequency of characters in enwik8 dataset: {freq}")

        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)
        # Convert to Unicode (preserving all characters in various languages)
        text = unicodedata.normalize("NFKC", text)  # Normalize to Unicode Compatibility
        # Keep only printable characters
        text = "".join(
            char if unicodedata.category(char) != "Cc" else " " for char in text
        )  # Replace control characters with spaces
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
        text = text.strip()

        with open(preprocessed_path, "w") as file:
            file.write(text)

    if verbose:
        print(f"Length of preprocessed enwik8 dataset: {len(text)}")
        print(f"Sample of preprocessed enwik8 dataset: {text[:10000]}")
        freq = sorted(Counter(text).items(), key=lambda item: item[1], reverse=True)
        print(f"Frequency of characters in preprocessed enwik8 dataset: {freq}")
    return text


def encode_enwik8(
    text: str, seq_len: int, percentage: float = 1
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Dict[int, str], Optional[str]]:
    """
    Encode the enwik8 dataset into sequences of characters.

    Args:
        text: The enwik8 text.
        seq_len: The length of each sequence.
        percentage: The percentage of the text to be used.

    Return:
        Encoded inputs, targets, character to index mapping,
        index to character mapping, padding token.
    """

    text = text[: int(len(text) * percentage)]
    unique_chars = get_enwik8_unique_chars(text)
    return encode_text(text, seq_len, unique_chars, pad_token=None)


def get_enwik8_dataloaders(
    path: str,
    seq_len: int,
    batch_size: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
    enwik8_percentage: float = 1,
) -> Tuple[
    Dict[str, int], Dict[int, str], Optional[str], DataLoader, DataLoader, DataLoader
]:
    """
    Get the data loaders for the enwik8 dataset.

    Args:
        path: The directory where the dataset resides.
        seq_len: The length of each sequence.
        batch_size: The batch size.
        train_percentage: The percentage of the dataset to be used for training.
        val_percentage: The percentage of the dataset to be used for validation.
        test_percentage: The percentage of the dataset to be used for testing.
        enwik8_percentage: The percentage of the enwik8 dataset to be used.

    Return:
        The character-to-index mapping, index-to-character mapping, padding token,
        and training, validation, and test data loaders.
    """

    text = load_enwik8(path)
    inputs, targets, char_to_idx, idx_to_char, pad_token = encode_enwik8(
        text, seq_len, enwik8_percentage
    )
    return (
        char_to_idx,
        idx_to_char,
        pad_token,
        *split_dataset(
            inputs,
            targets,
            batch_size,
            train_percentage,
            val_percentage,
            test_percentage,
        ),
    )
