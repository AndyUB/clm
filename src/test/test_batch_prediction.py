import torch
import os
import json
import time
import random
from argparse import Namespace, ArgumentParser
from typing import Dict, List, Tuple

from model.gpt2 import load_gpt2_model, gpt2_predict
from main.util import gpt2_inference as gpt2_batch_predict


def gpt2_single_predict(
    model_path: str,
    input_path: str,
    output_path: str,
    k: int,
    verbose: bool = True,
) -> None:
    """
    Generate predictions using the trained GPT-2 model one-at-a-time.

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
        predictions = []
        for sentence in sentences:
            predictions_probs = gpt2_predict(
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
            predictions.append(
                "".join(prediction for prediction, _ in predictions_probs)
            )
        file.write("\n".join(predictions) + "\n")


def generate_prefixes(
    count: int, length: Tuple[int, int], vocab: str, output_path: str
) -> List[str]:
    """
    Generate a list of random prefixes.

    Args:
        count: The number of prefixes to generate.
        length: The range of lengths for the prefixes.
        vocab: The vocabulary to sample characters from.
        output_path: The path to save the prefixes.

    Returns:
        A list of random prefixes.
    """

    prefixes = []
    for _ in range(count):
        prefix_length = random.randint(*length)
        prefix = "".join(random.choice(vocab) for _ in range(prefix_length))
        prefixes.append(prefix)

    with open(output_path, "w") as file:
        file.write("\n".join(prefixes) + "\n")

    return prefixes


def main(args: Namespace) -> None:
    """
    Compare the time difference between generating predictions one-at-a-time and
    in batch.

    Args:
        args: The command-line arguments.
    """

    VOCAB = "abcdefghijklmnopqrstuvwxyz "
    NUM_PREFIXES = 16000

    input_file = f"{args.min_len}_{args.max_len}.txt"
    input_path = os.path.join(args.input_dir, input_file)
    if not os.path.exists(input_path):
        generate_prefixes(NUM_PREFIXES, (args.min_len, args.max_len), VOCAB, input_path)

    batch_start = time.perf_counter()
    gpt2_batch_predict(
        args.model_path,
        input_path,
        args.output_path,
        args.k,
    )
    batch_end = time.perf_counter()
    batch_time = batch_end - batch_start

    single_start = time.perf_counter()
    gpt2_single_predict(
        args.model_path,
        input_path,
        args.output_path,
        args.k,
    )
    single_end = time.perf_counter()
    single_time = single_end - single_start

    print(f"Batch time: {batch_time:.2f}s")
    print(f"Single time: {single_time:.2f}s")
    print(f"Speedup: {single_time / batch_time:.2f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Compare batch and single predictions.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the GPT-2 model.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the predictions.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="k for top-k predictions.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        required=True,
        help="Minimum length of the prefix.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        required=True,
        help="Maximum length of the prefix.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
