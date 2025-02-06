import os
import random
import torch
import requests
import tarfile
import pandas as pd
from torch.utils.data import DataLoader

from data.util import encode_text, split_dataset
from typing import Tuple, Dict, List, Optional

SEED = 42
random.seed(SEED)


def load_tatoeba(path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Download the tatoeba dataset if it doesn't exist.

    Args:
        path: The directory where the dataset will be downloaded.
        verbose: Whether to print a summary of the loaded dataset.

    Return:
        The tatoeba dataset as a data frame.
    """

    url = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
    archive_name = "sentences.tar.bz2"
    extract_name = "extracted"
    archive_path = os.path.join(path, archive_name)
    extract_path = os.path.join(path, extract_name)

    if not os.path.exists(extract_path):
        os.makedirs(path, exist_ok=True)
        with requests.get(url, stream=True) as res:
            res.raise_for_status()
            with open(archive_path, "wb") as file:
                for chunk in res.iter_content(chunk_size=8192):
                    file.write(chunk)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(extract_path)

    csv_name = "sentences.csv"
    csv_path = os.path.join(extract_path, csv_name)
    df = pd.read_csv(csv_path, sep="\t", header=None, names=["id", "lang", "sentence"])

    if verbose:
        print(f"Number of sentences in the tatoeba dataset: {len(df)}")
        lang_counts = df.groupby("lang")["sentence"].count().reset_index()
        lang_counts = lang_counts.sort_values(by="sentence", ascending=False)
        print(f"Total number of languages: {lang_counts.shape[0]}")
        for _, row in lang_counts.iterrows():
            print(f"{row['lang']}: {row['sentence']}")
        print(df.head())

    return df


def get_tatoeba_unique_chars(sentences: List[str]) -> str:
    """
    Get the unique characters in the tatoeba dataset.

    Args:
        sentences: The tatoeba sentences.

    Return:
        The unique characters in the tatoeba dataset.
    """

    return sorted(set("".join(sentences)))


def encode_tatoeba(
    sentences: List[str], seq_len: int, percentage: float = 1
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Dict[int, str], Optional[str]]:
    """
    Encode the tatoeba dataset into sequences of characters.

    Args:
        sentences: The tatoeba sentences.
        seq_len: The length of each sequence.
        percentage: The percentage of the text to be used.
        seed: The random seed.

    Return:
        Encoded inputs, targets, character to index mapping,
        index to character mapping, padding token.
    """

    sample_size = int(len(sentences) * percentage)
    sentences = random.sample(sentences, sample_size)
    unique_chars = get_tatoeba_unique_chars(sentences)
    return encode_text(sentences, seq_len, unique_chars)


def get_tatoeba_dataloaders(
    path: str,
    seq_len: int,
    batch_size: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
    tatoeba_percentage: float = 1,
) -> Tuple[Dict[str, int], Dict[int, str], Optional[str], DataLoader, DataLoader, DataLoader]:
    """
    Get the data loaders for the tatoeba dataset.

    Args:
        path: The directory where the dataset resides.
        seq_len: The length of each sequence.
        batch_size: The batch size.
        train_percentage: The percentage of the dataset to be used for training.
        val_percentage: The percentage of the dataset to be used for validation.
        test_percentage: The percentage of the dataset to be used for testing.
        tatoeba_percentage: The percentage of the tatoeba dataset to be used.

    Return:
        The character-to-index mapping, index-to-character mapping, padding token,
        and training, validation, and test data loaders.
    """

    df = load_tatoeba(path)
    sentences = df["sentence"].tolist()
    inputs, targets, char_to_idx, idx_to_char, pad_token = encode_tatoeba(
        sentences, seq_len, tatoeba_percentage
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
