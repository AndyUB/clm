import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Tuple, Dict, List, Optional

from model.util import vectorize_text, preprocess_text


class GPT2CrossEntropyLoss(nn.Module):
    """
    A wrapper around nn.CrossEntropyLoss for the GPT-2 model.
    """

    def __init__(self):
        super(GPT2CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, outputs: CausalLMOutputWithCrossAttentions, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cross-entropy loss.

        Args:
            outputs: The model's outputs, which include the logits.
            targets: The target values.

        Return:
            The cross-entropy loss.
        """

        logits = outputs.logits
        loss: torch.Tensor = self.criterion(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        return loss


def gpt2_accuracy(
    outputs: CausalLMOutputWithCrossAttentions,
    targets: torch.Tensor,
    k: int,
) -> Tuple[int, int, int]:
    """
    Compute the accuracy of the GPT-2 model.

    Args:
        outputs: The model's outputs, which include the logits.
        targets: The target values.
        k: k for top-k accuracy.

    Return:
        The number of correct predictions, the number of correct top-k predictions,
        and the number of samples.
    """

    logits = outputs.logits
    _, topk_preds = logits.topk(k, dim=-1, sorted=True)
    top1_preds = topk_preds[:, :, 0]
    correct = (top1_preds == targets).sum().item()
    topk_correct = (
        topk_preds.eq(targets.unsqueeze(-1).expand_as(topk_preds)).sum().item()
    )
    total = targets.numel()
    return correct, topk_correct, total


def gpt2_predict(
    model: nn.Module,
    prefix: str,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    k: int,
    device: torch.device,
    pad_token: Optional[str] = None,
    seq_len: Optional[int] = None,
    lowercase: bool = False,
    remove_unknown: bool = True,
) -> List[Tuple[str, float]]:
    """
    Generate predictions using the GPT-2 model.

    Args:
        model: The GPT-2 model.
        prefix: The prefix to generate the predictions from.
        char_to_idx: The character to index mapping.
        idx_to_char: The index to character mapping.
        k: k for top-k predictions.
        device: The device to run the model on.
        pad_token: The padding token. If None, no padding is used. The padding token
            won't be included in the predictions.
        seq_len: The length of the sequence. If the prefix is longer than the
            sequence length, it is truncated. If None, the full sequence is used.
        lowercase: Whether to convert the input to lowercase.
        remove_unknown: Whether to remove unknown characters from the input.

    Return:
        A list of top-k predictions, each containing a character and its probability.
    """

    # Assume that the vocabulary size without the padding token is at least k
    if pad_token is None:
        padded_k = k
    else:
        padded_k = k + 1

    model.eval()
    with torch.no_grad():
        try:
            input_idxs = vectorize_text(
                prefix,
                char_to_idx,
                device,
                seq_len=seq_len,
                lowercase=lowercase,
                remove_unknown=remove_unknown,
            )
        except ValueError as e:
            print(e)
            return [("?", 1 / k)] * k

        # Get logits and probabilities
        outputs: CausalLMOutputWithCrossAttentions = model(input_idxs)
        logits = outputs.logits
        probs = torch.softmax(logits[0, -1], dim=-1)

        # Get top-k predictions
        top_k = torch.topk(probs, padded_k)
        top_k_indices = top_k.indices.tolist()
        top_k_probs = top_k.values.tolist()

        predictions = [
            (idx_to_char[idx], prob) for idx, prob in zip(top_k_indices, top_k_probs)
        ]
        if pad_token is not None:
            predictions = [
                (char, prob) for char, prob in predictions if char != pad_token
            ]
            predictions = predictions[: padded_k - 1]
        return predictions


def gpt2_batch_predict(
    model: nn.Module,
    prefixes: List[str],
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    k: int,
    device: torch.device,
    pad_token: Optional[str] = None,
    seq_len: Optional[int] = None,
    lowercase: bool = False,
    remove_unknown: bool = True,
    pred_batch_size: int = 128,
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
        pad_token: The padding token. If None, no padding is used. The padding token
            won't be included in the predictions.
        seq_len: The length of the sequence. If the prefix is longer than the
            sequence length, it is truncated. If None, the full sequence is used.
        lowercase: Whether to convert the input to lowercase.
        remove_unknown: Whether to remove unknown characters from the input.
        pred_batch_size: The batch size for predictions.

    Return:
        A list of concatenated top-k predictions for each prefix.
    """

    prefixes = [
        preprocess_text(
            prefix,
            char_to_idx,
            seq_len,
            lowercase,
            remove_unknown,
            suppress_error=True,
        )
        for prefix in prefixes
    ]
    len_to_prefixes: Dict[int, List[Tuple[str, int]]] = {}
    for i, prefix in enumerate(prefixes):
        prefix_len = len(prefix)
        if prefix_len not in len_to_prefixes:
            len_to_prefixes[prefix_len] = []
        len_to_prefixes[prefix_len].append((prefix, i))

    padded_k = k + 1 if pad_token is not None else k
    pad_index = char_to_idx[pad_token] if pad_token is not None else -1
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
                top_k = torch.topk(probs, padded_k, dim=-1)
                top_k_indices = top_k.indices.tolist()
                unpadded_predictions = [
                    "".join(idx_to_char[idx] for idx in indices if idx != pad_index)[:k]
                    for indices in top_k_indices
                ]

                for (_, index), prediction in zip(
                    batch_prefixes_indices, unpadded_predictions
                ):
                    predictions[index] = prediction

    return predictions


def load_gpt2_model(
    model_path: str, vocab_size: int, device: torch.device
) -> nn.Module:
    """
    Load the GPT-2 model on the specified device.

    Args:
        model_path: The path to the saved model.
        vocab_size: The size of the vocabulary.
        device: The device to run the model on.

    Return:
        The loaded GPT-2 model.
    """

    model: nn.Module = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model
