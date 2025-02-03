import os
import json
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from argparse import ArgumentParser, Namespace
from typing import Dict

from data.enwik8 import get_enwik8_dataloaders
from model.gpt2 import (
    GPT2CrossEntropyLoss,
    gpt2_accuracy,
    load_gpt2_model,
    gpt2_predict,
)
from model.util import train_model, evaluate_model


def train_evaluate(
    data_path: str, output_path: str, k: int, verbose: bool = True
) -> None:
    """
    Train and evaluate GPT-2 on the enwik8 dataset.

    Args:
        data_path: The path to the enwik8 dataset.
        output_path: The path to save the model and mappings.
        k: k for top-k accuracy.
        verbose: Whether to print outputs.
    """

    SEQ_LEN = 50
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_percentage = 0.8
    val_percentage = 0.1
    test_percentage = 0.1
    enwik8_percentage = 0.1

    # Load the data
    if verbose:
        print("Loading data...")
    char_to_idx, _, train_loader, val_loader, test_loader = get_enwik8_dataloaders(
        data_path,
        SEQ_LEN,
        BATCH_SIZE,
        train_percentage,
        val_percentage,
        test_percentage,
        enwik8_percentage,
    )
    vocab_size = len(char_to_idx)
    if verbose:
        print(f"Vocabulary size: {vocab_size}")

    model: nn.Module = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(vocab_size)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = GPT2CrossEntropyLoss()

    # Train the model
    if verbose:
        print("Training model...")
    model = train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        DEVICE,
        EPOCHS,
    )

    # Save the model
    if verbose:
        print("Saving model...")
    output_dir = os.path.join(output_path, f"gpt2_{enwik8_percentage}enwik8")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/model.pth")
    with open(f"{output_dir}/char_to_idx.json", "w") as file:
        json.dump(char_to_idx, file)

    # Evaluate the model
    if verbose:
        print("Evaluating model...")
    evaluate_model(model, val_loader, criterion, gpt2_accuracy, DEVICE, k)
    evaluate_model(model, test_loader, criterion, gpt2_accuracy, DEVICE, k)


def predict(
    model_path: str, input_path: str, output_path: str, k: int, verbose: bool = True
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
    idx_to_char: Dict[int, str] = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(char_to_idx)
    model = load_gpt2_model(
        os.path.join(model_path, "model.pth"),
        vocab_size,
        DEVICE,
    )

    if verbose:
        print(f"Vocabulary size: {vocab_size}")
        print(f"char_to_idx: {char_to_idx}")
        print("Generating predictions...")
    with open(input_path, "r") as file:
        sentences = file.readlines()
    with open(output_path, "w") as file:
        for sentence in sentences:
            predictions = gpt2_predict(
                model,
                sentence,
                char_to_idx,
                idx_to_char,
                k,
                DEVICE,
                SEQ_LEN,
                lowercase=True,
                remove_unknown=True,
            )
            file.write("".join(char for char, _ in predictions) + "\n")


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
        description="Train and evaluate GPT-2 on enwik8 dataset, or generate predictions."
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


def main(args: Namespace):
    """
    Train and evaluate GPT-2 on enwik8 dataset, or generate predictions.

    Args:
        args: Arguments parsed from the command line.
    """

    if args.mode == "train":
        train_evaluate(args.data_path, args.output_path, args.k)
    elif args.mode == "predict":
        predict(args.model_path, args.input_path, args.output_path, args.k)
    else:
        raise NotImplementedError(f"Mode {args.mode} is not implemented.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
