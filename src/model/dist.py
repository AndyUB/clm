import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Callable, Tuple, Any

from data.dist import LengthGroupedBatchDataset
from main.util import HyperparameterConfig


def save_checkpoint(
    model: DDP,
    optimizer: optim.Optimizer,
    epoch: int,
    checkpoint_path: str,
    verbose: bool = True,
) -> None:
    """
    Save a checkpoint of the model.

    Args:
        model: The model to save, wrapped in DDP.
        optimizer: The optimizer to save.
        epoch: The current epoch.
        checkpoint_path: The path to save the checkpoint.
        verbose: Whether to print messages.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    if verbose:
        print(f"Checkpoint saved at: {checkpoint_path}")


def load_checkpoint(
    model: DDP,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    rank: int,
    verbose: bool = True,
) -> int:
    """
    Load a checkpoint for the model and optimizer.

    Args:
        model: The model to load the checkpoint into, wrapped in DDP.
        optimizer: The optimizer to load the checkpoint into.
        checkpoint_path: The path to the checkpoint.
        rank: The process rank.
        verbose: Whether to print messages.

    Returns:
        The epoch to resume training from.
    """

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{rank}")
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if verbose:
            print(
                f"[rank {rank}] Checkpoint loaded. Resuming from epoch {start_epoch}."
            )
    else:
        start_epoch = 0
        if verbose:
            print(f"[rank {rank}] No checkpoint found. Starting from scratch.")
    return start_epoch


def distributed_evaluate_model(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: Callable[[Any, torch.Tensor, int], Tuple[int, int, int]],
    k: int,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model: The model to evaluate.
        val_dataloader: The validation dataloader.
        loss_fn: The loss function to use.
        accuracy_fn: The accuracy function to use.
        device: The device to run the model on.
        verbose: Whether to print detailed information.

    Returns:
        The average loss, accuracy, and top-3 accuracy.
    """

    model.eval()
    total_loss = 0
    total_correct = 0
    total_topk_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch: torch.Tensor
            batch = batch.to(device)
            inputs = batch[:, :-1].contiguous()
            labels = batch[:, 1:].contiguous()
            if verbose:
                print(f"inputs shape: {inputs.shape}")
                print(f"labels shape: {labels.shape}")

            outputs = model(inputs)
            loss: torch.Tensor = loss_fn(outputs, labels)
            correct, topk_correct, samples = accuracy_fn(outputs, labels, k)

            total_loss += loss.item()
            total_correct += correct
            total_topk_correct += topk_correct
            total_samples += samples

    return (
        total_loss / len(val_dataloader),
        total_correct / total_samples,
        total_topk_correct / total_samples,
    )


def distributed_train_model(
    rank: int,
    world_size: int,
    load_model_fn: Callable[[], nn.Module],
    train_dataset: LengthGroupedBatchDataset,
    val_dataset: LengthGroupedBatchDataset,
    hyperparameters: HyperparameterConfig,
    criterion: nn.Module,
    accuracy_fn: Callable[[Any, torch.Tensor, int], Tuple[int, int, int]],
    k: int,
    checkpoint_dir: str,
    checkpoint_file: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    report_interval: int = 10,
    eval_result_dir: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Function that runs on each process to train GPT-2 using DDP.

    Args:
        rank: The rank of the current process.
        world_size: The total number of processes.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        load_model_fn: The function to load the model.
        hyperparameters: The hyperparameters to use.
        criterion: The loss function to use.
        accuracy_fn: The accuracy function to use.
        k: The value of k for top-k accuracy.
        checkpoint_dir: The directory to save checkpoints.
        checkpoint_file: The checkpoint file to load.
            If None, training starts from scratch.
        checkpoint_interval: The number of epochs between checkpoints.
            If None, no checkpoints are saved.
        report_interval: The number of batches between time reports.
        eval_result_dir: The directory to save evaluation results.
            If None, evaluation is skipped.
        verbose: Whether to print detailed messages.
    """

    batch_size = hyperparameters.batch_size
    sequence_length = hyperparameters.seq_len
    learning_rate = hyperparameters.lr
    num_epochs = hyperparameters.epochs

    # Set up environment variables for DDP
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "16384"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if verbose:
        print(f"Initialized process group: rank {rank}, world_size {world_size}")
    device = torch.device(f"cuda:{rank}")
    model = load_model_fn()
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    if verbose:
        print(f"Model initialized on rank {rank}")

    if rank == 0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if checkpoint_file is None:
        start_epoch = 0
        if verbose:
            print(f"[rank {rank}] No checkpoint file provided. Training from scratch.")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, rank, verbose)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=LengthGroupedBatchDataset.collate,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=LengthGroupedBatchDataset.collate,
    )

    train_losses, val_losses, val_accuracies, val_topk_accuracies = [], [], [], []

    num_batches = len(train_dataloader)
    total_batches = num_epochs * num_batches
    batch_count = 0
    start_time = time.perf_counter()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # Ensures randomness across epochs
        train_sampler.set_epoch(epoch)

        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            batch: torch.Tensor
            batch = batch.to(device)
            inputs = batch[:, :-1].contiguous()
            labels = batch[:, 1:].contiguous()

            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if verbose and rank == 0:
                if batch_count % report_interval == 0:
                    elapsed_time = time.perf_counter() - start_time
                    avg_time_per_batch = elapsed_time / batch_count
                    remaining_time = avg_time_per_batch * (total_batches - batch_count)
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Batch [{i + 1}/{num_batches}], "
                        f"Loss: {loss.item():.4f}, "
                        f"Elapsed time: {elapsed_time:.2f} s, "
                        f"Remaining time: {remaining_time:.2f} s, "
                        f"Progress: {100 * batch_count / total_batches:.2f}%"
                    )

        train_losses.append(epoch_loss / num_batches)

        if eval_result_dir and rank == 0:
            val_loss, val_acc, val_topk_acc = distributed_evaluate_model(
                model,
                val_dataloader,
                criterion,
                accuracy_fn,
                k,
                device,
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_topk_accuracies.append(val_topk_acc)

            if verbose:
                print(
                    f"Epoch {epoch + 1} Complete: "
                    f"Train loss: {train_losses[-1]:.4f}, "
                    f"Val loss: {val_loss:.4f}, "
                    f"Val accuracy: {val_acc:.4f}, "
                    f"Val top-{k} accuracy: {val_topk_acc:.4f}"
                )

        if (
            checkpoint_interval is not None
            and rank == 0
            and (epoch + 1) % checkpoint_interval == 0
            and epoch != num_epochs - 1
        ):
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(checkpoint_dir, f"epoch{epoch}.pt"),
                verbose,
            )

    if rank == 0:
        save_checkpoint(
            model,
            optimizer,
            num_epochs,
            os.path.join(checkpoint_dir, f"final_{num_epochs}epochs.pt"),
            verbose,
        )

        if eval_result_dir is not None:
            os.makedirs(eval_result_dir, exist_ok=True)
            losses_figure_path = os.path.join(
                eval_result_dir, f"loss-{num_epochs}epochs.png"
            )
            accs_figure_path = os.path.join(
                eval_result_dir, f"accuracy-{num_epochs}epochs.png"
            )
            csv_path = os.path.join(eval_result_dir, f"{num_epochs}epochs.csv")

            actual_epochs = list(range(start_epoch + 1, num_epochs + 1))
            plt.figure()
            plt.plot(actual_epochs, train_losses, marker="o", label="Train Loss")
            plt.plot(actual_epochs, val_losses, marker="s", label="Validation Loss")
            for i in range(len(actual_epochs)):
                plt.annotate(
                    f"{train_losses[i]:.2f}",
                    (actual_epochs[i], train_losses[i]),
                    textcoords="offset points",
                    xytext=(0, -12),
                    ha='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )
                plt.annotate(
                    f"{val_losses[i]:.2f}",  # Text for val_topk_accuracies
                    (actual_epochs[i], val_losses[i]),  # Position at the data point
                    textcoords="offset points",
                    xytext=(0, 8),  # Offset text slightly below the point
                    ha='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(
                f"Losses over {num_epochs} epochs "
                f"(resumed from epoch {start_epoch})"
            )
            plt.savefig(losses_figure_path)
            plt.close()

            plt.figure()
            plt.plot(
                actual_epochs, val_accuracies, marker="o", label="Validation Accuracy"
            )
            plt.plot(
                actual_epochs,
                val_topk_accuracies,
                marker="s",
                label=f"Validation Top-{k} Accuracy",
            )
            for i in range(len(actual_epochs)):
                plt.annotate(
                    f"{val_accuracies[i]:.2f}",
                    (actual_epochs[i], val_accuracies[i]),
                    textcoords="offset points",
                    xytext=(0, -12),
                    ha='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )
                plt.annotate(
                    f"{val_topk_accuracies[i]:.2f}",  # Text for val_topk_accuracies
                    (actual_epochs[i], val_topk_accuracies[i]),  # Position at the data point
                    textcoords="offset points",
                    xytext=(0, 8),  # Offset text slightly below the point
                    ha='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
                )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title(
                f"Accuracies over {num_epochs} epochs "
                f"(resumed from epoch {start_epoch})"
            )
            plt.savefig(accs_figure_path)
            plt.close()
            if verbose:
                print(
                    "Saved evaluation metrics plot at: "
                    f"{losses_figure_path} and {accs_figure_path}"
                )

            actual_epochs_trained = num_epochs - start_epoch
            df = pd.DataFrame(
                {
                    "sequence_length": [sequence_length] * actual_epochs_trained,
                    "batch_size": [batch_size] * actual_epochs_trained,
                    "learning_rate": [learning_rate] * actual_epochs_trained,
                    "epoch": actual_epochs,
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                    "val_accuracy": val_accuracies,
                    "val_topk_accuracy": val_topk_accuracies,
                }
            )
            df.to_csv(csv_path, index=False)
            if verbose:
                print(f"Saved evaluation metrics CSV at: {csv_path}")

    dist.destroy_process_group()
    if verbose:
        print(f"Destroyed process group: rank {rank}")
