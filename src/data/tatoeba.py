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


def read_tatoeba_csv(path: str) -> pd.DataFrame:
    """
    Read the tatoeba dataset from a csv file.

    Args:
        path: The path to the csv file.

    Return:
        The tatoeba dataset as a data frame.
    """

    return pd.read_csv(path, sep="\t", header=None, names=["id", "lang", "sentence"])


def load_tatoeba(
    path: str,
    percentage: float = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Download the tatoeba dataset if it doesn't exist.

    Args:
        path: The directory where the dataset will be downloaded.
        percentage: The percentage of the dataset to be used.
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

    csv_name = "sentences_preprocessed.csv"
    if percentage == 1:
        prefixed_csv_name = csv_name
    else:
        prefixed_csv_name = f"{percentage}{csv_name}"
    prefixed_csv_path = os.path.join(extract_path, prefixed_csv_name)

    if not os.path.exists(prefixed_csv_path):
        csv_path = os.path.join(extract_path, csv_name)
        if not os.path.exists(csv_path):
            csv_without_duplicates_name = "sentences_unique.csv"
            csv_without_duplicates_path = os.path.join(extract_path, csv_without_duplicates_name)
            if not os.path.exists(csv_without_duplicates_path):
                csv_with_duplicates_name = "sentences.csv"
                csv_with_duplicates_path = os.path.join(extract_path, csv_with_duplicates_name)
                df = read_tatoeba_csv(csv_with_duplicates_path)
                df = df.drop_duplicates(subset=["sentence"])
                df.to_csv(csv_without_duplicates_path, sep="\t", index=False, header=False)
            else:
                df = read_tatoeba_csv(csv_without_duplicates_path)
            df = df[df['sentence'].str.len() > 1]  # Keep sentences with length > 1
            df.to_csv(csv_path, sep="\t", index=False, header=False)
        else:
            df = read_tatoeba_csv(csv_path)
        df = df.sample(frac=percentage)
        df.to_csv(prefixed_csv_path, sep="\t", index=False, header=False)
    else:
        df = read_tatoeba_csv(prefixed_csv_path)

    if verbose:
        print(f"Number of sentences in the tatoeba dataset: {len(df)}")
        length_to_count = dict()
        for sentence in df["sentence"].tolist():
            length_to_count[len(sentence)] = length_to_count.get(len(sentence), 0) + 1
        print("Sentence length distribution")
        for length, count in length_to_count.items():
            print(f"Length: {length}, Count: {count}")
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
) -> Tuple[
    Dict[str, int], Dict[int, str], Optional[str], DataLoader, DataLoader, DataLoader
]:
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

    df = load_tatoeba(path, percentage=tatoeba_percentage)
    sentences = df["sentence"].tolist()
    inputs, targets, char_to_idx, idx_to_char, pad_token = encode_tatoeba(
        sentences, seq_len
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
