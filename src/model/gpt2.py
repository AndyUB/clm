import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Tuple, Dict, List, Optional

from model.util import preprocess_input


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
    if pad_token is not None:
        k += 1

    if seq_len is not None:
        prefix = prefix[-seq_len:]

    model.eval()
    with torch.no_grad():
        input_idxs = preprocess_input(
            prefix, char_to_idx, device, lowercase, remove_unknown
        )

        # Get logits and probabilities
        outputs: CausalLMOutputWithCrossAttentions = model(input_idxs)
        logits = outputs.logits
        probs = torch.softmax(logits[0, -1], dim=-1)

        # Get top-k predictions
        top_k = torch.topk(probs, k)
        top_k_indices = top_k.indices.tolist()
        top_k_probs = top_k.values.tolist()

        predictions = [
            (idx_to_char[idx], prob) for idx, prob in zip(top_k_indices, top_k_probs)
        ]
        if pad_token is not None:
            predictions = [(char, prob) for char, prob in predictions if char != pad_token]
            predictions = predictions[:k - 1]
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
