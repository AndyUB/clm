import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoModelForCausalLM
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional

from model.gpt2 import (
    GPT2CrossEntropyLoss,
    gpt2_accuracy,
    load_gpt2_model,
    gpt2_predict,
)
from model.util import train_model, evaluate_model


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
        description="Train and evaluate GPT-2 on text8 dataset, or generate predictions."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        required=True,
        help="Mode: train or predict",
    )
    parser.add_argument("--k", type=int, required=True, help="k for top-k predictions")
    parser.add_argument("--output-path", type=str, help="Path to save outputs")

    # Train mode arguments
    parser.add_argument("--data-path", type=str, help="Path to training data")

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
):
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
    evaluate_model(model, val_loader, criterion, gpt2_accuracy, device, k)
    if verbose:
        print("Evaluating model on test set...")
    evaluate_model(model, test_loader, criterion, gpt2_accuracy, device, k)


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
        for sentence in sentences:
            sentence = sentence[:-1]  # Remove newline character
            predictions = gpt2_predict(
                model,
                sentence,
                char_to_idx,
                idx_to_char,
                k,
                DEVICE,
                pad_token=pad_token,
                seq_len=SEQ_LEN,
                lowercase=True,
                remove_unknown=True,
            )
            file.write("".join(char for char, _ in predictions) + "\n")
            if verbose:
                print(f"{sentence=}, {predictions=}")
