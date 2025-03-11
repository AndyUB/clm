import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import List, Dict, Tuple


def gpt2_batch_predict_o(
    model: nn.Module,
    prefixes: List[str],
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    k: int,
    device: torch.device,
    seq_len: int,
    pred_batch_size: int,
) -> List[str]:
    """
    Generate predictions for multiple prefixes using the GPT-2 model.

    Args:
        model: The GPT-2 model.
        prefixes: The prefixes to generate the predictions from.
        char_to_idx: The character to index mapping.
        idx_to_char: The index to character mapping.
        k: k for top-k predictions.
        device: The device to run the model on.
        seq_len: The length of the sequence. If the prefix is longer than the
            sequence length, it is truncated.
        pred_batch_size: The batch size for predictions.

    Return:
        A list of concatenated top-k predictions for each prefix.
    """
    processed_prefixes = []
    for prefix in prefixes:
        filtered_prefix = "".join(char for char in prefix if char in char_to_idx)
        truncated_prefix = filtered_prefix[-seq_len:]
        processed_prefixes.append(truncated_prefix)

    len_to_prefixes: Dict[int, List[Tuple[str, int]]] = {}
    for i, prefix in enumerate(processed_prefixes):
        prefix_len = len(prefix)
        if prefix_len not in len_to_prefixes:
            len_to_prefixes[prefix_len] = []
        len_to_prefixes[prefix_len].append((prefix, i))

    unknown_prediction = "?" * k
    predictions = [unknown_prediction for _ in prefixes]
    model.eval()
    with torch.no_grad():
        for prefix_len, prefixes_indices in len_to_prefixes.items():
            if prefix_len == 0:
                continue

            num_prefixes = len(prefixes_indices)
            num_batches = (num_prefixes + pred_batch_size - 1) // pred_batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * pred_batch_size
                end_idx = min((batch_idx + 1) * pred_batch_size, num_prefixes)
                batch_prefixes_indices = prefixes_indices[start_idx:end_idx]

                try:
                    vectorized_prefixes = torch.stack(
                        [
                            torch.tensor(
                                [char_to_idx[char] for char in prefix],
                                dtype=torch.long,
                                device=device,
                            )
                            for prefix, _ in batch_prefixes_indices
                        ]
                    )
                    outputs: CausalLMOutputWithCrossAttentions = model(vectorized_prefixes)
                    logits = outputs.logits
                    probs = torch.softmax(logits[:, -1], dim=-1)
                    top_k = torch.topk(probs, k, dim=-1)
                    top_k_indices = top_k.indices.tolist()
                except Exception:
                    continue

                batch_predictions = [
                    "".join(idx_to_char[idx] for idx in indices)
                    for indices in top_k_indices
                ]

                for (_, index), prediction in zip(
                    batch_prefixes_indices, batch_predictions
                ):
                    predictions[index] = prediction

    return predictions


def gpt2_inference_o(
    model_path: str,
    input_path: str,
    output_path: str,
    k: int,
    verbose: bool = True,
) -> None:
    """
    Generate predictions with the trained GPT-2 model. The model has no padding token.

    Args:
        model_path: The path to the saved model's directory.
        input_path: The path to the input data for inference.
        output_path: The path to save the predictions.
        k: k for top-k predictions.
        verbose: Whether to print outputs.
    """

    SEQ_LEN = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file = os.path.join(model_path, "model.pt")
    char_to_idx_file = os.path.join(model_path, "char_to_idx.json")
    if verbose:
        print("Loading model...")
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

    sentences = [
        sentence[:-1] for sentence in sentences
    ]  # Remove newline character
    if verbose:
        print("Generating predictions...")
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

