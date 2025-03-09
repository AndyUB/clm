import os
import json
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import List, Optional
import torch.multiprocessing as mp
import torch.nn as nn
import transformers

from data.dist import get_distributed_tatoeba_datasets
from model.dist import distributed_train_model
from model.gpt2 import GPT2CrossEntropyLoss, gpt2_accuracy
from main.util import HyperparameterConfig, hyperparam_combinations


def search_space_for_gpt2_on_tatoeba_distributed() -> List[HyperparameterConfig]:
    """
    Get the hyperparameter search space for distributed training of GPT-2 on the
    tatoeba dataset.

    Returns:
        The hyperparameter search space.
    """

    # seq_lens = [50, 60, 70, 80, 90, 100]
    # seq_lens = [50, 100, 150, 200]
    seq_lens = [60, 80]
    # batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # batch_sizes = [4, 16, 64, 256]
    batch_sizes = [4, 16, 64]
    # lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # lrs = [1e-6, 1e-4, 1e-2]
    lrs = [1e-6, 1e-4]
    # epochs = [5, 10, 15, 20]
    # epochs = [5, 15, 25]
    epochs = [10, 20]
    return hyperparam_combinations(
        seq_lens=seq_lens, batch_sizes=batch_sizes, lrs=lrs, epochs=epochs
    )


def load_gpt2_model(vocab_size: int) -> nn.Module:
    """
    Load the GPT-2 model.

    Args:
        vocab_size: The size of the vocabulary.

    Returns:
        The GPT-2 model, resized to the given vocabulary size.
    """

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(vocab_size)
    return model


def distributed_train_gpt2_on_tatoeba(
    world_size: int,
    train_dataset: int,
    val_dataset: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    epochs: int,
    k: int,
    checkpoint_dir: str,
    checkpoint_file: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    report_interval: int = 10,
    eval_result_dir: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Train GPT-2 on the tatoeba dataset using multiple GPUs.

    Args:
        world_size: The number of GPUs in distributed training.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        seq_len: The length of each sequence.
        batch_size: The desired batch size.
        lr: The learning rate.
        epochs: The number of epochs.
        k: The k for top-k accuracy.
        checkpoint_dir: The directory where the checkpoints will be saved.
        checkpoint_file: The checkpoint file to load the model from.
        checkpoint_interval: Number of epochs between checkpoints.
        report_interval: Number of batches between reports.
        eval_result_dir: The directory to save the evaluation results.
        verbose: Whether to log verbose information.
    """

    mp.spawn(
        distributed_train_model,
        args=(
            world_size,
            partial(load_gpt2_model, vocab_size),
            train_dataset,
            val_dataset,
            HyperparameterConfig(
                seq_len=seq_len,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
            ),
            GPT2CrossEntropyLoss(),
            gpt2_accuracy,
            k,
            checkpoint_dir,
            checkpoint_file,
            checkpoint_interval,
            report_interval,
            eval_result_dir,
            verbose,
        ),
        nprocs=world_size,
        join=True,
    )


def distributed_tune_gpt2_on_tatoeba(
    search_space: List[HyperparameterConfig],
    data_dir: str,
    data_percentage: float,
    world_size: int,
    k: int,
    checkpoint_dir: str,
    eval_result_dir: str,
    verbose: bool = True,
) -> None:
    """
    Tune GPT-2 on the tatoeba dataset using multiple GPUs.

    Args:
        search_space: The hyperparameter search space.
        data_dir: The directory where the dataset resides.
        data_percentage: The percentage of the dataset to be used.
        world_size: The number of GPUs in distributed training.
        k: The k for top-k accuracy.
        checkpoint_dir: The directory where the checkpoints will be saved.
        eval_result_dir: The directory to save the evaluation results.
        verbose: Whether to log verbose information.
    """

    train_val_test_split = [0.8, 0.2, 0]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_result_dir, exist_ok=True)
    for config in search_space:
        train_dataset, _, char_to_idx, _, val_dataset, _ = (
            get_distributed_tatoeba_datasets(
                data_dir,
                data_percentage,
                train_val_test_split,
                config.seq_len,
                config.batch_size,
                world_size,
                include_non_full_batches=True,
            )
        )
        config_name = (
            f"{config.seq_len}seq_"
            f"{config.batch_size}batch_"
            f"{config.lr}lr_"
            f"{config.epochs}epochs"
        )
        model_dir = os.path.join(checkpoint_dir, config_name)
        result_dir = os.path.join(eval_result_dir, config_name)
        vocab_size = len(char_to_idx)
        if verbose:
            print(f"{config=}: {vocab_size=}")
        distributed_train_gpt2_on_tatoeba(
            world_size,
            train_dataset,
            val_dataset,
            vocab_size,
            config.seq_len,
            config.batch_size,
            config.lr,
            config.epochs,
            k,
            checkpoint_dir=model_dir,
            checkpoint_file=None,
            checkpoint_interval=None,
            report_interval=100,
            eval_result_dir=result_dir,
            verbose=verbose,
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Train or tune GPT-2 on the tatoeba dataset using multiple GPUs."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "tune"],
        required=True,
        help="Whether to train or tune the model.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="The number of workers in distributed training.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="The directory where the dataset resides.",
    )
    parser.add_argument(
        "--data-percentage",
        type=float,
        required=True,
        help="The percentage of the dataset to be used.",
    )
    parser.add_argument(
        "--include-non-full-batches",
        action="store_true",
        help="Whether to include non-full batches in the training dataset.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="The length of each sequence.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The desired batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="The learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="k for top-k accuracy.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="The directory where the checkpoints will be saved.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="The checkpoint file to load the model from.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Number of epochs between checkpoints.",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=10,
        help="Number of batches between reports.",
    )
    parser.add_argument(
        "--eval-result-dir",
        type=str,
        default=None,
        help="The directory to save the evaluation results",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to turn off verbose logging.",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    if args.mode == "train":
        train_val_test_split = [0.8, 0.2, 0]
        train_dataset, _, char_to_idx, _, val_dataset, _ = (
            get_distributed_tatoeba_datasets(
                args.data_dir,
                args.data_percentage,
                train_val_test_split,
                args.seq_len,
                args.batch_size,
                args.world_size,
                args.include_non_full_batches,
            )
        )
        vocab_size = len(char_to_idx)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(args.checkpoint_dir, "char_to_idx.json"), "w") as file:
            json.dump(char_to_idx, file)
        distributed_train_gpt2_on_tatoeba(
            args.world_size,
            train_dataset,
            val_dataset,
            vocab_size,
            args.seq_len,
            args.batch_size,
            args.lr,
            args.epochs,
            args.k,
            args.checkpoint_dir,
            args.checkpoint_file,
            args.checkpoint_interval,
            args.report_interval,
            args.eval_result_dir,
            not args.silent,
        )
    elif args.mode == "tune":
        if args.eval_result_dir is None:
            raise ValueError("eval_result_dir is required for tuning.")
        distributed_tune_gpt2_on_tatoeba(
            search_space=search_space_for_gpt2_on_tatoeba_distributed(),
            data_dir=args.data_dir,
            data_percentage=args.data_percentage,
            world_size=args.world_size,
            k=args.k,
            checkpoint_dir=args.checkpoint_dir,
            eval_result_dir=args.eval_result_dir,
            verbose=not args.silent,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main(parse_args())
