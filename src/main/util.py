import os
import json
import pandas as pd
import torch
from dataclasses import dataclass, astuple
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModelForCausalLM
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Tuple, Callable, List

from model.gpt2 import (
    GPT2CrossEntropyLoss,
    gpt2_accuracy,
    load_gpt2_model,
    gpt2_batch_predict,
)
from model.util import train_model, evaluate_model


@dataclass
class HyperparameterConfig:
    """
    Configuration for hyperparameters.

    Attributes:
        seq_len: The sequence length.
        batch_size: The batch size.
        lr: The learning rate.
        epochs: The number of epochs.
    """

    seq_len: int
    batch_size: int
    lr: float
    epochs: int

    @staticmethod
    def hyperparam_names() -> List[str]:
        """
        Get the names of the hyperparameters.

        Return:
            The list of hyperparameter names.
        """
        return ["seq_len", "batch_size", "lr", "epochs"]


def parse_args() -> Namespace:
    """
    Parse command-line arguments. Supports train and predict modes.
    Both modes require k for top-k predictions.
    Train mode also requires data_path and output_path.
    Predict mode requires model_path, input_path, and output_path.

    Return:
        The parsed arguments.
    """

    parser = ArgumentParser(
        description="Train and evaluate GPT-2 on text8 dataset, "
        "generate predictions, or search hyperparameters."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "hyperparam"],
        required=True,
        help="Mode: train, predict, or hyperparameter search",
    )
    parser.add_argument(
        "--k", type=int, required=False, default=3, help="k for top-k predictions"
    )
    parser.add_argument("--output-path", type=str, help="Path to save outputs")

    # Train / hyperparameter mode arguments
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument(
        "--data-percentage",
        type=float,
        default=0.05,
        help="Percentage of dataset to use",
    )

    # Predict mode arguments
    parser.add_argument("--model-path", type=str, help="Path to saved model")
    parser.add_argument(
        "--input-path", type=str, help="Path to input data for inference"
    )

    args = parser.parse_args()

    # Validate required arguments based on mode
    if args.mode == "train":
        if not args.data_path or not args.output_path:
            parser.error("Train mode requires --data-path and --output-path.")
    elif args.mode == "hyperparam":
        if not args.data_path or not args.output_path:
            parser.error(
                "Hyperparameter search mode requires --data-path and --output-path."
            )
    else:
        assert args.mode == "predict"
        if not args.model_path or not args.input_path or not args.output_path:
            parser.error(
                "Test mode requires --model-path, --input-path, and --output-path."
            )

    return args


def gpt2_train_eval(
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    output_path: str,
    model_name: str,
    char_to_idx: Dict[str, int],
    k: int,
    device: torch.device,
    pad_token: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """
    Train and evaluate the GPT-2 model on the given dataset.

    Args:
        vocab_size: The size of the vocabulary.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        test_loader: The test data loader.
        epochs: The number of epochs to train the model.
        learning_rate: The learning rate for training.
        output_path: The path to save the trained model.
        model_name: The name of the model.
        char_to_idx: The mapping from characters to indices.
        k: k for top-k predictions.
        device: The device to use for training.
        pad_token: The padding token.
        verbose: Whether to print outputs.

    Returns:
        The validation and test losses, accuracies, and top-k accuracies.
    """

    model: nn.Module = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(vocab_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = GPT2CrossEntropyLoss()

    # Train the model
    if verbose:
        print("Training model...")
    model = train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        epochs,
    )

    # Save the model
    if verbose:
        print("Saving model...")
    output_dir = os.path.join(output_path, model_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/model.pth")
    with open(f"{output_dir}/char_to_idx.json", "w") as file:
        json.dump(char_to_idx, file)
    if pad_token is not None:
        with open(f"{output_dir}/pad_token.txt", "w") as file:
            file.write(pad_token)

    # Evaluate the model
    if verbose:
        print("Evaluating model on validation set...")
    val_loss, val_acc, val_topk_acc = evaluate_model(
        model, val_loader, criterion, gpt2_accuracy, device, k
    )
    if verbose:
        print("Evaluating model on test set...")
    test_loss, test_acc, test_topk_acc = evaluate_model(
        model, test_loader, criterion, gpt2_accuracy, device, k
    )
    return val_loss, val_acc, val_topk_acc, test_loss, test_acc, test_topk_acc


def gpt2_inference(
    model_path: str,
    input_path: str,
    output_path: str,
    k: int,
    verbose: bool = True,
) -> None:
    """
    Generate predictions using the trained GPT-2 model.

    Args:
        model_path: The path to the saved model.
        input_path: The path to the input data for inference.
        output_path: The path to save the predictions.
        k: k for top-k predictions.
        verbose: Whether to print outputs.
    """

    SEQ_LEN = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("Loading model...")
    with open(os.path.join(model_path, "char_to_idx.json"), "r") as file:
        char_to_idx: Dict[str, int] = json.load(file)
    pad_token = None
    pad_token_path = os.path.join(model_path, "pad_token.txt")
    if os.path.exists(pad_token_path):
        with open(pad_token_path, "r") as file:
            pad_token = file.read().strip()
    idx_to_char: Dict[int, str] = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(char_to_idx)
    model = load_gpt2_model(
        os.path.join(model_path, "model.pth"),
        vocab_size,
        DEVICE,
    )

    if verbose:
        print("Generating predictions...")
    with open(input_path, "r") as file:
        sentences = file.readlines()
    with open(output_path, "w") as file:
        sentences = [
            sentence[:-1] for sentence in sentences
        ]  # Remove newline character
        predictions = gpt2_batch_predict(
            model,
            sentences,
            char_to_idx,
            idx_to_char,
            k,
            DEVICE,
            pad_token=pad_token,
            seq_len=SEQ_LEN,
            lowercase=True,
            remove_unknown=True,
        )
        file.write("\n".join(predictions) + "\n")


def hyperparam_combinations(
    seq_lens: List[int],
    batch_sizes: List[int],
    lrs: List[float],
    epochs: List[int],
) -> List[HyperparameterConfig]:
    """
    Generate all combinations of hyperparameters.

    Args:
        seq_lens: The sequence lengths.
        batch_sizes: The batch sizes.
        lrs: The learning rates.
        epochs: The numbers of epochs.

    Returns:
        The list of hyperparameter configurations.
    """

    return [
        HyperparameterConfig(seq_len, batch_size, lr, epoch)
        for seq_len in seq_lens
        for batch_size in batch_sizes
        for lr in lrs
        for epoch in epochs
    ]


def gpt2_hyperparam_search(
    train_val_fn: Callable[
        [str, float, str, int, HyperparameterConfig], Tuple[float, float, float]
    ],
    search_space: List[HyperparameterConfig],
    dataset_name: str,
    data_path: str,
    data_percentage: float,
    output_path: str,
    k: int,
    verbose: bool = True,
) -> None:
    """
    Search for good hyperparameters for the GPT-2 model on the given dataset.

    Args:
        train_val_fn: The function to train the model and evaluate it on the
            validation set. It takes the data path, data percentage, output
            path, k, and hyperparameters and returns the validation loss,
            accuracy, and top-k accuracy.
        search_space: The combinations of hyperparameters to search.
        dataset_name: The name of the dataset.
        data_path: The path to the dataset.
        output_path: The path to save the model and mappings.
        k: k for top-k accuracy.
        verbose: Whether to print outputs.
        data_percentage: The percentage of the dataset to use.
        verbose: Whether to print outputs.
    """

    search_results = []
    for config in search_space:
        val_loss, val_acc, val_topk_acc = train_val_fn(
            data_path, data_percentage, output_path, k, config
        )
        search_results.append((*astuple(config), val_loss, val_acc, val_topk_acc))
        if verbose:
            print(
                f"Hyperparameters: {config}, "
                f"Validation loss: {val_loss}, "
                f"Validation accuracy: {val_acc}, "
                f"Validation top-{k} accuracy: {val_topk_acc}"
            )

    COLUMN_NAMES = [
        *HyperparameterConfig.hyperparam_names(),
        "val_avg_loss",
        "val_acc",
        "val_topk_acc",
    ]
    df = pd.DataFrame(data=search_results, columns=COLUMN_NAMES)
    csv_path = os.path.join(
        output_path, f"{data_percentage}{dataset_name}_hyperparam_search.csv"
    )
    df.to_csv(csv_path, index=False)
