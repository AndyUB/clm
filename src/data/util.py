import os
import requests
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Union, Optional


class TextDataset(Dataset):
    """
    A dataset for text data. Each sample is a pair of inputs and targets.
    """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Initialize the text dataset.

        Args:
            inputs: The input tensors.
            targets: The target tensors.
        """

        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def download_unzip(url: str, path: str, zip_name: str) -> None:
    """
    Download and unzip specified file if it doesn't exist.

    Args:
        url: The URL of the file to be downloaded.
        path: The directory where the file will be downloaded and unzipped.
        zip_name: The zip file name without the .zip extension.
    """

    text_path = os.path.join(path, zip_name)
    zip_path = f"{text_path}.zip"

    # Download the file if it doesn't exist
    if not os.path.exists(text_path):
        os.makedirs(path, exist_ok=True)
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path)


def encode_text(
    text: Union[str, List[str]],
    seq_len: int,
    unique_chars: str,
    pad_token: Optional[str] = "[PAD]",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Divide the text into sequences then encode each sequence into inputs and targets.

    Args:
        text: The text to be encoded. It can be a long string or a list of sentences.
        seq_len: The length of each sequence.
        unique_chars: The unique characters in the text.

    Return:
        inputs: The input tensors.
        targets: The target tensors.
        char_to_idx: A dictionary that maps each character to its index.
        idx_to_char: A dictionary that maps each index to its character.
        pad_token: The padding token. If None, no padding is used.
    """

    if isinstance(text, str):
        text = [text]

    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    if pad_token is not None:
        char_to_idx[pad_token] = len(char_to_idx)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    inputs = []
    targets = []
    for sentence in text:
        encoded_sentence = torch.tensor(
            [char_to_idx[char] for char in sentence if char in char_to_idx],
            dtype=torch.long,
        )
        if pad_token is not None and len(encoded_sentence) % (seq_len + 1) != 0:
            if len(encoded_sentence) < seq_len + 1:
                # Sentence is too short, pad it
                padding = torch.tensor(
                    [char_to_idx[pad_token]] * (seq_len + 1 - len(encoded_sentence)),
                    dtype=torch.long,
                )
                encoded_sentence = torch.cat([encoded_sentence, padding])
            else:
                # Use the last (seq_len + 1) characters as the last sequence
                padding = encoded_sentence[-(seq_len + 1) :]
                encoded_sentence = torch.cat(
                    [
                        encoded_sentence[
                            : len(encoded_sentence) // (seq_len + 1) * (seq_len + 1)
                        ],
                        padding,
                    ]
                )
        num_sequences = len(encoded_sentence) // (seq_len + 1)
        for i in range(num_sequences):
            start = i * (seq_len + 1)
            end = start + seq_len + 1
            sequence = encoded_sentence[start:end]
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])

    return (
        torch.stack(inputs),
        torch.stack(targets),
        char_to_idx,
        idx_to_char,
        pad_token,
    )


def split_dataset(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        inputs: The input tensors.
        targets: The target tensors.
        batch_size: The batch size.
        train_percentage: The percentage of the dataset to be used for training.
        val_percentage: The percentage of the dataset to be used for validation.
        test_percentage: The percentage of the dataset to be used for testing.

    Return:
        train_loader: The training data loader.
        val_loader: The validation data loader.
        test_loader: The test data loader.
    """

    if train_percentage + val_percentage + test_percentage != 1:
        raise ValueError(
            "The sum of train_percentage, val_percentage, and test_percentage must be 1."
        )
    dataset_size = len(inputs)
    train_size = int(train_percentage * dataset_size)
    val_size = int(val_percentage * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_inputs, val_inputs, test_inputs = torch.split(
        inputs, [train_size, val_size, test_size]
    )
    train_targets, val_targets, test_targets = torch.split(
        targets, [train_size, val_size, test_size]
    )

    train_dataset = TextDataset(train_inputs, train_targets)
    val_dataset = TextDataset(val_inputs, val_targets)
    test_dataset = TextDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader
