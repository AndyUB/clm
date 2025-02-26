import torch
import pandas as pd
from argparse import Namespace

from data.text8 import get_text8_dataloaders
from main.util import parse_args, gpt2_train_eval, gpt2_inference


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


def hyperparam_search(
    data_path: str, output_path: str, k: int, data_percentage=0.0001, verbose=True
):
    """
    Train and evaluate GPT-2 on the text8 dataset with many different hyperparameters.
    Log metadata about hyperparameters and save to CSV.

    Args:
        data_path: The path to the text8 dataset.
        output_path: The path to save the model and mappings.
        k: k for top-k accuracy.
        verbose: Whether to print outputs.
    """

    COLUMN_NAMES = [
        "dataset",
        "data_percentage",
        "seq_len",
        "batch_size",
        "lr",
        "epochs",
        "val_avg_loss",
        "val_acc",
        "val_topk_acc",
        "test_avg_loss",
        "test_acc",
        "test_topk_acc",
    ]
    dlist = []
    for seq_len in range(50, 60, 10):
        for batch_size in [2**i for i in range(6, 7)]:
            for lr in [10**j for j in range(-6, -5)]:
                for epochs in range(5, 10, 5):
                    if verbose:
                        print(
                            f"Training GPT2 on {data_percentage*100}% Text8: {seq_len} seq, {batch_size} batch, {lr} learning rate, {epochs} epochs"
                        )
                    (
                        val_avg_loss,
                        val_acc,
                        val_topk_acc,
                        test_avg_loss,
                        test_acc,
                        test_topk_acc,
                    ) = train_gpt2_on_text8(
                        data_path,
                        output_path,
                        k,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        lr=lr,
                        epochs=epochs,
                        data_percentage=data_percentage,
                    )
                    data = [
                        "text8",
                        data_percentage,
                        seq_len,
                        batch_size,
                        lr,
                        epochs,
                        val_avg_loss,
                        val_acc,
                        val_topk_acc,
                        test_avg_loss,
                        test_acc,
                        test_topk_acc,
                    ]
                    dlist.append(dict(zip(COLUMN_NAMES, data)))
    df = pd.DataFrame(dlist)
    df.to_csv(output_path + "/gpt2_text8_hparamsearch.csv", index=False)


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
        hyperparam_search(
            args.data_path,
            args.output_path,
            args.k,
            data_percentage=args.data_percentage,
        )
    else:
        raise NotImplementedError(f"Mode {args.mode} is not implemented.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
