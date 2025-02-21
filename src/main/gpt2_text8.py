import torch
from argparse import Namespace

from data.text8 import get_text8_dataloaders
from main.util import parse_args, gpt2_train_eval, gpt2_inference


def train_gpt2_on_text8(
    data_path: str, output_path: str, k: int, verbose: bool = True
) -> None:
    """
    Train and evaluate GPT-2 on the text8 dataset.

    Args:
        data_path: The path to the text8 dataset.
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
    text8_percentage = 0.1

    # Load the data
    if verbose:
        print("Loading data...")
    char_to_idx, _, pad_token, train_loader, val_loader, test_loader = (
        get_text8_dataloaders(
            data_path,
            SEQ_LEN,
            BATCH_SIZE,
            train_percentage,
            val_percentage,
            test_percentage,
            text8_percentage,
        )
    )
    vocab_size = len(char_to_idx)

    gpt2_train_eval(
        vocab_size,
        train_loader,
        val_loader,
        test_loader,
        EPOCHS,
        LEARNING_RATE,
        output_path,
        f"gpt2_{text8_percentage}text8",
        char_to_idx,
        k,
        DEVICE,
        pad_token=pad_token,
        verbose=verbose,
    )


def main(args: Namespace):
    """
    Train and evaluate GPT-2 on text8 dataset, or generate predictions.

    Args:
        args: Arguments parsed from the command line.
    """

    if args.mode == "train":
        train_gpt2_on_text8(args.data_path, args.output_path, args.k)
    elif args.mode == "predict":
        gpt2_inference(args.model_path, args.input_path, args.output_path, args.k)
    else:
        raise NotImplementedError(f"Mode {args.mode} is not implemented.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
