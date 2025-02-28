import torch
from argparse import Namespace
from typing import Tuple, List

from data.text8 import get_text8_dataloaders
from main.util import (
    parse_args,
    gpt2_train_eval,
    gpt2_inference,
    gpt2_hyperparam_search,
    HyperparameterConfig,
    hyperparam_combinations,
)


def train_gpt2_on_text8(
    data_path: str,
    output_path: str,
    k: int,
    seq_len=50,
    batch_size=64,
    epochs=5,
    lr=1e-4,
    data_percentage=0.001,
    verbose: bool = True,
):
    """
    Train and evaluate GPT-2 on the text8 dataset.

    Args:
        data_path: The path to the text8 dataset.
        output_path: The path to save the model and mappings.
        k: k for top-k accuracy.
        seq_len: number of tokens of context the transformer uses
        batch_size: number of training examples to process at a time before loss and backprop
        epochs: number of times to pass all data through
        lr: learning rate of training (multiply by gradient to modify weights)
        data_percentage: percentage of text8 dataset to use
        verbose: Whether to print outputs.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_percentage = 0.8
    val_percentage = 0.1
    test_percentage = 0.1

    # Load the data
    if verbose:
        print("Loading data...")
    char_to_idx, _, pad_token, train_loader, val_loader, test_loader = (
        get_text8_dataloaders(
            data_path,
            seq_len,
            batch_size,
            train_percentage,
            val_percentage,
            test_percentage,
            data_percentage,
        )
    )
    vocab_size = len(char_to_idx)

    return gpt2_train_eval(
        vocab_size,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        lr,
        output_path,
        f"gpt2_{data_percentage}text8_{seq_len}seq_{batch_size}batch_{lr}lr_{epochs}eps",
        char_to_idx,
        k,
        DEVICE,
        pad_token=pad_token,
        verbose=verbose,
    )


def tune_gpt2_on_text8(
    data_path: str,
    data_percentage: float,
    output_path: str,
    k: int,
    hyperparameters: HyperparameterConfig,
) -> Tuple[float, float, float]:
    """
    Tune GPT-2 on the text8 dataset.

    Args:
        data_path: The path to the text8 dataset.
        data_percentage: Percentage of text8 dataset to use.
        output_path: The path to save the model and mappings.
        k: k for top-k accuracy.
        hyperparameters: The hyperparameters to use.

    Returns:
        Validation average loss, accuracy, and top-k accuracy.
    """

    val_loss, val_acc, val_topk_acc, _, _, _ = train_gpt2_on_text8(
        data_path,
        output_path,
        k,
        seq_len=hyperparameters.seq_len,
        batch_size=hyperparameters.batch_size,
        epochs=hyperparameters.epochs,
        lr=hyperparameters.lr,
        data_percentage=data_percentage,
    )
    return val_loss, val_acc, val_topk_acc


def search_space_for_gpt2_on_text8() -> List[HyperparameterConfig]:
    """
    Get the hyperparameter search space for training GPT-2 on the text8 dataset.

    Returns:
        The hyperparameter search space.
    """

    # seq_lens = [50, 100, 150, 200]
    seq_lens = [50, 100]
    # batch_sizes = [4, 8, 16, 32, 64, 128, 256]
    batch_sizes = [4, 16]
    # lrs = [1e-5, 1e-4, 1e-3, 1e-2]
    lrs = [1e-4, 1e-2]
    # epochs = [5, 10, 15, 20]
    epochs = [10, 20]
    return hyperparam_combinations(
        seq_lens=seq_lens, batch_sizes=batch_sizes, lrs=lrs, epochs=epochs
    )


def main(args: Namespace):
    """
    Train and evaluate GPT-2 on text8 dataset, or generate predictions.

    Args:
        args: Arguments parsed from the command line.
    """

    if args.mode == "train":
        train_gpt2_on_text8(
            args.data_path,
            args.output_path,
            args.k,
            data_percentage=args.data_percentage,
        )
    elif args.mode == "predict":
        gpt2_inference(args.model_path, args.input_path, args.output_path, args.k)
    elif args.mode == "hyperparam":
        gpt2_hyperparam_search(
            train_val_fn=tune_gpt2_on_text8,
            search_space=search_space_for_gpt2_on_text8(),
            dataset_name="text8",
            data_path=args.data_path,
            data_percentage=args.data_percentage,
            output_path=args.output_path,
            k=args.k,
        )
    else:
        raise NotImplementedError(f"Mode {args.mode} is not implemented.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
