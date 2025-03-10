import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from data.tatoeba import load_tatoeba, get_tatoeba_unique_chars, read_tatoeba_csv


def sentences_to_sequences(
    sentences: List[str],
    seq_len: int,
    char_to_idx: Dict[str, int],
    extend_tail: bool = True,
) -> List[List[int]]:
    """
    Convert a list of sentences to a list of sequences. A sequence is a list of
    indices, where each index uniquely represents a character. If a sentence is
    shorter than the given sequence length plus 1, it will be kept as is.

    Args:
        sentences: List of sentences.
        seq_len: Length of each sequence. During training, each sequence contains
            seq_len + 1 characters, where the first seq_len characters are inputs
            and the last seq_len characters are targets.
        char_to_idx: Mapping of characters to indices.
        extend_tail: Whether to extend the last sequence backwards if it is short.

    Returns:
        List of sequences.
    """

    train_seq_len = 1 + seq_len
    sequences: List[List[int]] = []
    for sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len <= train_seq_len:
            sequences.append([char_to_idx[char] for char in sentence])
            continue
        num_sequences = (sentence_len + train_seq_len - 1) // train_seq_len
        for i in range(num_sequences):
            start = i * train_seq_len
            end = min(start + train_seq_len, sentence_len)
            if end - start < train_seq_len and extend_tail:
                start = end - train_seq_len
            sequence = [char_to_idx[char] for char in sentence[start:end]]
            sequences.append(sequence)
    return sequences


def sequences_to_batches(
    sequences: List[List[int]],
    batch_size: int,
    world_size: int,
    shuffle: bool = False,
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    """
    Put sequences of the same length into one group. Split each group into
    batches of size that is no larger than the specified batch size. We create
    two bundles of batches:
    1. In the first bundle, we gather n * W full batches together for the largest
    possible n, where W is the given world size and n means the number of batches
    per worker.
    2. The remaining full batches and all non-full batches are gathered in the
    second bundle.

    Args:
        sequences: List of sequences. Each sequence is a list of indices, where
            each index uniquely represents a character.
        batch_size: Desired batch size.
        world_size: Number of workers (W).
        shuffle: Whether to shuffle the batches in each bundle.

    Returns:
        full_bundle: The bundle of W * n full batches, for the largest possible n.
        non_full_bundle: The bundle of remaining full batches (< W) and non-full
            batches (<= sequence length).
    """

    length_to_group: Dict[int, List[List[int]]] = defaultdict(list)
    for sequence in sequences:
        length_to_group[len(sequence)].append(sequence)

    full_batches: List[List[List[int]]] = []
    non_full_batches: List[List[List[int]]] = []

    for _, group in length_to_group.items():
        num_batches = (len(group) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(group))
            batch = group[start:end]
            if len(batch) == batch_size:
                full_batches.append(batch)
            else:
                non_full_batches.append(batch)

    num_full_batches = len(full_batches)
    full_bundle_size = (num_full_batches // world_size) * world_size
    full_bundle = full_batches[:full_bundle_size]
    leftover_full_batches = full_batches[full_bundle_size:]
    non_full_bundle = non_full_batches
    non_full_bundle.extend(leftover_full_batches)

    if shuffle:
        random.shuffle(full_bundle)
        random.shuffle(non_full_bundle)

    return full_bundle, non_full_bundle


class LengthGroupedBatchDataset(Dataset):
    def __init__(self, batches: List[List[List[int]]]):
        """
        Store batches of sequences. All sequences in a batch have the same length.

        Args:
            batches: List of batches.
        """

        self.batches = batches

    def __len__(self) -> int:
        """
        Get the number of batches in the dataset.

        Returns:
            Number of batches.
        """

        return len(self.batches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Fetch a batch.

        Args:
            idx: Index of the batch.

        Returns:
            The batch as a tensor.
        """

        return torch.tensor(self.batches[idx], dtype=torch.long)

    @staticmethod
    def collate(batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Custom collate function for DataLoader.

        Args:
            batch: List containing 1 tensor, which is a batch of sequences.

        Returns:
            The batch of sequences as a tensor.
        """

        return batch[0]

    def restore(self, idx_to_char: Dict[int, str]) -> Dict[str, int]:
        """
        Restore the batches in this dataset into sequences of text.

        Args:
            idx_to_char: Index-to-character mapping.

        Returns:
            A dictionary from a restored sequence to its count, i.e.,
            the number of times that same restored sequence appears in
            this dataset.
        """

        sequence_to_count = dict()
        for batch in self.batches:
            for sequence in batch:
                sequence = "".join(idx_to_char[idx] for idx in sequence)
                sequence_to_count[sequence] = sequence_to_count.get(sequence, 0) + 1
        return sequence_to_count


def batches_to_dataset(
    train_bundle: List[List[List[int]]],
    leftover_bundle: List[List[List[int]]],
) -> LengthGroupedBatchDataset:
    """
    Convert the bundles of batches to datasets.

    Args:
        train_bundle: The bundle of batches for training.
        leftover_bundle: The bundle of leftover batches.

    Returns:
        A dataset for training and a dataset for leftover batches.
    """

    train_dataset = LengthGroupedBatchDataset(train_bundle)
    leftover_dataset = LengthGroupedBatchDataset(leftover_bundle)

    return train_dataset, leftover_dataset


def sentences_to_train_dataset(
    sentences: List[str],
    seq_len: int,
    batch_size: int,
    world_size: int,
    include_non_full_batches: bool = True,
    leftover_sequences: Dict[str, int] = None,
) -> Tuple[
    LengthGroupedBatchDataset,
    LengthGroupedBatchDataset,
    Dict[str, int],
    Dict[int, str],
]:
    """
    Convert a list of sentences to a dataset for training.

    Args:
        sentences: List of sentences.
        seq_len: Length of each sequence.
        batch_size: Desired batch size.
        world_size: Number of workers.
        include_non_full_batches: Whether to include non-full batches in the
            training dataset.
        leftover_sequences: A dictionary from each leftover sequence to its count.

    Returns:
        The training dataset, the leftover dataset, the character-to-index mapping,
        and the index-to-character mapping.
    """

    unique_chars = get_tatoeba_unique_chars(sentences)
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    sequences = sentences_to_sequences(sentences, seq_len, char_to_idx)
    if leftover_sequences is not None:
        filtered_sequences = []
        for sequence in sequences:
            sequence_as_text = "".join(idx_to_char[idx] for idx in sequence)
            if leftover_sequences.get(sequence_as_text, 0) > 0:
                leftover_sequences[sequence_as_text] -= 1
            else:
                filtered_sequences.append(sequence)
        sequences = filtered_sequences
    full_batches, non_full_batches = sequences_to_batches(
        sequences, batch_size, world_size
    )
    if include_non_full_batches:
        extended_size = len(non_full_batches) // world_size * world_size
        full_batches.extend(non_full_batches[:extended_size])
        non_full_batches = non_full_batches[extended_size:]
    if leftover_sequences is not None and len(full_batches) % world_size != 0:
        raise ValueError(
            "Incorrect leftover sentences are provided. "
            f"The number of full batches recovered is {len(full_batches)}, "
            f"not divisible by the number of workers {world_size}"
        )
    train_dataset, leftover_dataset = batches_to_dataset(
        full_batches,
        non_full_batches,
    )
    return train_dataset, leftover_dataset, char_to_idx, idx_to_char


def csv_to_eval_dataset(
    eval_pct: float,
    path: str,
    seq_len: int,
    eval_batch_size: int,
    char_to_idx: Dict[str, int],
) -> Optional[LengthGroupedBatchDataset]:
    """
    Convert a CSV file to an evaluation dataset.

    Args:
        eval_pct: The percentage of evaluation data in the training-validation-test
            split.
        path: The path to the CSV file.
        seq_len: The length of each sequence.
        eval_batch_size: The evaluation batch size.
        char_to_idx: The character-to-index mapping.
    """

    if eval_pct == 0:
        return None

    df = read_tatoeba_csv(path)
    return df_to_eval_dataset(df, seq_len, eval_batch_size, char_to_idx)


def df_to_eval_dataset(
    df: pd.DataFrame,
    seq_len: int,
    eval_batch_size: int,
    char_to_idx: Dict[str, int],
) -> LengthGroupedBatchDataset:
    """
    Convert a DataFrame to an evaluation dataset.

    Args:
        df: The DataFrame.
        seq_len: The length of each sequence.
        eval_batch_size: The evaluation batch size.
        char_to_idx: The character-to-index mapping.

    Returns:
        The evaluation dataset.
    """

    sentences = df["sentence"].tolist()
    sentences = [
        "".join(char for char in sentence if char in char_to_idx)
        for sentence in sentences
    ]
    sentences = [
        sentence for sentence in sentences if len(sentence) > 1
    ]
    sequences = sentences_to_sequences(sentences, seq_len, char_to_idx)
    full_batches, non_full_batches = sequences_to_batches(
        sequences, eval_batch_size, world_size=1
    )
    batches = full_batches + non_full_batches
    return LengthGroupedBatchDataset(batches)


def get_distributed_tatoeba_datasets(
    data_dir: str,
    data_percentage: float,
    train_val_test_split: Tuple[float, float, float],
    seq_len: int,
    batch_size: int,
    world_size: int,
    include_non_full_batches: bool = True,
    verbose: bool = True,
) -> Tuple[
    LengthGroupedBatchDataset,
    LengthGroupedBatchDataset,
    Dict[str, int],
    Dict[int, str],
    LengthGroupedBatchDataset,
    LengthGroupedBatchDataset,
]:
    """
    Fetch Tatoeba datasets for distributed training.

    Args:
        path: The directory where the dataset resides."
        data_percentage: The percentage of the Tatoeba dataset to be used.
        train_val_test_split: The split ratio for training, validation, and testing.
        seq_len: The length of each sequence.
        batch_size: The desired batch size.
        world_size: The number of workers in distributed training.
        include_non_full_batches: Whether to include non-full batches in the
            training dataset.
        verbose: Whether to log detailed messages.

    Returns:
        The training dataset that can be evenly distributed among workers, the
        leftover dataset, the character-to-index mapping, the index-to-character
        mapping, the validation dataset, and the test dataset.
    """

    if verbose:
        print(
            f"Generating datasets for {data_percentage}% of tatoeba: "
            f"{seq_len=}, {batch_size=}, {world_size=}, {train_val_test_split=}, "
            f"{include_non_full_batches=}"
        )

    train_pct, val_pct, test_pct = train_val_test_split
    if train_pct <= 0:
        raise ValueError("The training split ratio must be positive.")
    if val_pct < 0 or test_pct < 0:
        raise ValueError("The validation and test split ratios must be non-negative.")
    if train_pct + val_pct + test_pct != 1:
        raise ValueError("The sum of the split ratios must be 1.")

    data_pct_dir = os.path.join(data_dir, str(data_percentage))
    os.makedirs(data_pct_dir, exist_ok=True)
    train_val_test_dir = os.path.join(
        data_pct_dir, f"{train_pct}train_{val_pct}val_{test_pct}test"
    )
    train_path = os.path.join(train_val_test_dir, "train.csv")
    val_path = os.path.join(train_val_test_dir, "val.csv")
    test_path = os.path.join(train_val_test_dir, "test.csv")
    leftover_path = os.path.join(train_val_test_dir, "leftover.json")

    val_dataset = None
    test_dataset = None
    if os.path.exists(train_val_test_dir):
        train_df = read_tatoeba_csv(train_path)
        train_sentences = train_df["sentence"].tolist()
        with open(leftover_path, "r") as f:
            leftover_sequences = json.load(f)
        train_dataset, leftover_dataset, char_to_idx, idx_to_char = (
            sentences_to_train_dataset(
                train_sentences,
                seq_len,
                batch_size,
                world_size,
                include_non_full_batches,
                leftover_sequences,
            )
        )
        val_dataset = csv_to_eval_dataset(
            val_pct, val_path, seq_len, batch_size, char_to_idx
        )
        test_dataset = csv_to_eval_dataset(
            test_pct, test_path, seq_len, batch_size, char_to_idx
        )
    else:
        os.makedirs(train_val_test_dir, exist_ok=True)
        df: pd.DataFrame = load_tatoeba(data_dir, data_percentage)
        sentences = df.values.tolist()
        val_sentences = []
        test_sentences = []
        if train_pct == 1:
            train_sentences = sentences
        else:
            train_size = int(train_pct * len(sentences))
            train_sentences = sentences[:train_size]
            if test_pct == 0:
                val_sentences = sentences[train_size:]
            elif val_pct == 0:
                test_sentences = sentences[train_size:]
            else:
                val_size = int(val_pct * len(sentences))
                val_sentences = sentences[train_size : train_size + val_size]
                test_sentences = sentences[train_size + val_size :]

        train_df = pd.DataFrame(train_sentences, columns=df.columns)
        val_df = pd.DataFrame(val_sentences, columns=df.columns)
        test_df = pd.DataFrame(test_sentences, columns=df.columns)
        train_df.to_csv(train_path, sep="\t", index=False, header=False)
        val_df.to_csv(val_path, sep="\t", index=False, header=False)
        test_df.to_csv(test_path, sep="\t", index=False, header=False)

        train_sentences = train_df["sentence"].tolist()
        train_dataset, leftover_dataset, char_to_idx, idx_to_char = (
            sentences_to_train_dataset(
                train_sentences,
                seq_len,
                batch_size,
                world_size,
                include_non_full_batches,
            )
        )
        leftover_sequences = leftover_dataset.restore(idx_to_char)
        with open(leftover_path, "w") as f:
            json.dump(leftover_sequences, f)
        if val_pct > 0:
            val_dataset = df_to_eval_dataset(
                val_df,
                seq_len,
                batch_size,
                char_to_idx,
            )
        if test_pct > 0:
            test_dataset = df_to_eval_dataset(
                test_df,
                seq_len,
                batch_size,
                char_to_idx,
            )

    return (
        train_dataset,
        leftover_dataset,
        char_to_idx,
        idx_to_char,
        val_dataset,
        test_dataset,
    )
