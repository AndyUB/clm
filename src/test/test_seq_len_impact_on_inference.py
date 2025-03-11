import os
import json
import time
import torch
import torch.nn as nn
from argparse import Namespace, ArgumentParser
from transformers import AutoModelForCausalLM
from typing import Dict

from main.inference_o import gpt2_batch_predict_o
from main.inference_o import gpt2_inference_o as long_seq_len_inference
from test.test_batch_prediction import generate_prefixes


def short_seq_len_inference(
    model_path: str,
    input_path: str,
    output_path: str,
    k: int,
) -> None:
    """
    Generate predictions with the trained GPT-2 model. The model has no padding token.
    The sequence length is 50.

    Args:
        model_path: The path to the saved model's directory.
        input_path: The path to the input data for inference.
        output_path: The path to save the predictions.
        k: k for top-k predictions.
    """

    SEQ_LEN = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = os.path.join(model_path, "model.pt")
    char_to_idx_file = os.path.join(model_path, "char_to_idx.json")
    with open(char_to_idx_file, "r") as file:
        char_to_idx: Dict[str, int] = json.load(file)
    idx_to_char: Dict[int, str] = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    model: nn.Module = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(vocab_size)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.to(DEVICE)

    with open(input_path, "r") as file:
        sentences = file.readlines()

    sentences = [sentence[:-1] for sentence in sentences]  # Remove newline character
    predictions = gpt2_batch_predict_o(
        model,
        sentences,
        char_to_idx,
        idx_to_char,
        k,
        DEVICE,
        SEQ_LEN,
        pred_batch_size=128,
    )

    with open(output_path, "w") as file:
        file.write("\n".join(predictions) + "\n")


def main(args: Namespace) -> None:
    """
    Test sequence length impact on inference speed.

    Args:
        args: The command-line arguments.
    """

    VOCAB = "abcdefghijklmnopqrstuvwxyz "
    NUM_PREFIXES = 16000

    input_file = f"{args.min_len}_{args.max_len}.txt"
    input_path = os.path.join(args.input_dir, input_file)
    if not os.path.exists(input_path):
        generate_prefixes(NUM_PREFIXES, (args.min_len, args.max_len), VOCAB, input_path)

    torch.cuda.reset_peak_memory_stats()
    long_start = time.perf_counter()
    long_seq_len_inference(
        args.model_path,
        input_path,
        args.output_path,
        args.k,
    )
    long_end = time.perf_counter()
    long_time = long_end - long_start
    long_peak_memory = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    short_start = time.perf_counter()
    short_seq_len_inference(
        args.model_path,
        input_path,
        args.output_path,
        args.k,
    )
    short_end = time.perf_counter()
    short_time = short_end - short_start
    short_peak_memory = torch.cuda.max_memory_allocated()

    print(f"Long time: {long_time:.2f}s")
    print(f"Short time: {short_time:.2f}s")
    print(f"Slowdown: {long_time / short_time:.2f}")
    print()

    print(
        f"Long peak memory: {long_peak_memory / 1024 / 1024:.2f}MB "
        f"({long_peak_memory} bytes)"
    )
    print(
        f"Short peak memory: {short_peak_memory / 1024 / 1024:.2f}MB "
        f"({short_peak_memory} bytes)"
    )
    print(f"Memory increase: {short_peak_memory - long_peak_memory} bytes")


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
