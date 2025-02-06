import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from typing import Tuple, Callable, Any, Dict


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    verbose: bool = True,
) -> nn.Module:
    """
    Train the model.

    Args:
        model: The model to be trained.
        dataloader: The data loader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run the model on.
        epochs: The number of epochs.
        verbose: Whether to print the loss.

    Return:
        The trained model.
    """

    model.train()

    num_batches = len(dataloader)
    start_time = time.perf_counter()
    print(f"[{start_time}] # batches = {num_batches}")

    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if verbose and (i + 1) % 10 == 0:
                curr_time = time.perf_counter()
                elapsed_time = curr_time - start_time
                remaining_time = (
                    elapsed_time
                    / (epoch * num_batches + i + 1)
                    * ((epochs - epoch) * num_batches - i - 1)
                )
                print(
                    f"[{curr_time}] epoch {epoch + 1}/{epochs} "
                    f"batch {i + 1}/{num_batches}: "
                    f"elapse = {elapsed_time:.4f}s, "
                    f"remaining = {remaining_time:.4f}s"
                )
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
            )
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    accuracy_fn: Callable[[Any, torch.Tensor, int], Tuple[int, int, int]],
    device: torch.device,
    k: int = 3,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Evaluate the model.

    Args:
        model: The model to be evaluated.
        dataloader: The data loader.
        criterion: The loss function.
        accuracy_fn: The function to calculate accuracy.
            It returns the number of correct predictions,
            the number of top-k correct predictions,
            and the total number of samples.
        device: The device to run the model on.
        k: k for top-k accuracy.
        verbose: Whether to print the evaluation results.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    model.eval()
    total_loss = 0
    correct = 0
    topk_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)

            total_loss += loss.item()

            # Calculate accuracy
            correct_batch, topk_correct_batch, total_batch = accuracy_fn(
                outputs, targets, k
            )
            correct += correct_batch
            topk_correct += topk_correct_batch
            total += total_batch

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    topk_accuracy = topk_correct / total

    if verbose:
        print(f"Evaluation Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Top-{k} Accuracy: {topk_accuracy:.4f}")

    return avg_loss, accuracy, topk_accuracy


def preprocess_input(
    text: str,
    char_to_idx: Dict[str, int],
    device: torch.device,
    seq_len: int = None,
    lowercase: bool = False,
    remove_unknown: bool = True,
) -> torch.Tensor:
    """
    Convert the text to a tensor of indices.

    Args:
        text: The text to be processed.
        char_to_idx: The character-to-index mapping.
        lowercase: Whether to convert the text to lowercase.
        seq_len: The length of the sequence. If None, the full sequence is used.
        remove_unknown: Whether to remove unknown characters.
        device: The device the tensor resides on.

    Return:
        The tensor of indices.
    """

    if lowercase:
        text = text.lower()
    if remove_unknown:
        text = "".join(char for char in text if char in char_to_idx)
    if seq_len is not None:
        text = text[-seq_len:]

    if len(text) == 0:
        raise ValueError("[ERROR] Empty input prefix.")

    return (
        torch.tensor([char_to_idx[char] for char in text], dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )
