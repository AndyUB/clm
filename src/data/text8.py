import os
import torch
from torch.utils.data import DataLoader

from data.util import download_unzip, encode_text, split_dataset
from typing import Tuple, Dict


def load_text8(path: str) -> str:
    """
    Download and unzip the text8 dataset if it doesn't exist.

    Args:
        path: The directory where the dataset will be downloaded and unzipped.

    Return:
        Content of the text8 dataset.
    """

    url = "http://mattmahoney.net/dc/text8.zip"
    zip_name = "text8"
    text_path = os.path.join(path, zip_name)

    download_unzip(url, path, zip_name)

    # Read the text file
    with open(text_path, "r") as file:
        text = file.read()
    return text


def encode_text8(
    text: str, seq_len: int, percentage: float = 1
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Encode the text8 dataset into sequences of characters.

    Args:
        text: The text8 text.
        seq_len: The length of each sequence.
        percentage: The percentage of the text to be used.

    Return:
        Encoded inputs, targets, character to index mapping,
        and index to character mapping.
    """

    unique_chars = sorted("abcdefghijklmnopqrstuvwxyz ")
    text = text[: int(len(text) * percentage)]
    return encode_text(text, seq_len, unique_chars)


def get_text8_dataloaders(
    path: str,
    seq_len: int,
    batch_size: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
    text8_percentage: float = 1,
) -> Tuple[Dict[str, int], Dict[int, str], DataLoader, DataLoader, DataLoader]:
    """
    Get the data loaders for the text8 dataset.

    Args:
        path: The directory where the dataset resides.
        seq_len: The length of each sequence.
        batch_size: The batch size.
        train_percentage: The percentage of the dataset to be used for training.
        val_percentage: The percentage of the dataset to be used for validation.
        test_percentage: The percentage of the dataset to be used for testing.
        text8_percentage: The percentage of the text8 dataset to be used.

    Return:
        The character-to-index mapping, index-to-character mapping, and
        training, validation, and test data loaders.
    """

    text = load_text8(path)
    inputs, targets, char_to_idx, idx_to_char = encode_text8(
        text, seq_len, text8_percentage
    )
    return (
        char_to_idx,
        idx_to_char,
        *split_dataset(
            inputs,
            targets,
            batch_size,
            train_percentage,
            val_percentage,
            test_percentage,
        ),
    )
