import os
import json
import torch
from argparse import ArgumentParser, Namespace

from main.dist import load_gpt2_model

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Extract model states from checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory where the checkpoint file resides.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Checkpoint filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the output file containing extracted model states.",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    checkpoint_dir: str = args.checkpoint_dir
    checkpoint_file: str = args.checkpoint_file
    output_dir: str = args.output_dir
    char_to_idx_file = "char_to_idx.json"

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")
    char_to_idx_path = os.path.join(checkpoint_dir, char_to_idx_file)
    if not os.path.exists(char_to_idx_path):
        raise ValueError(f"{char_to_idx_path} does not exist")

    with open(char_to_idx_path, "r") as f:
        char_to_idx = json.load(f)
    vocab_size = len(char_to_idx)
    print(f"Loaded {char_to_idx_file}, vocab size: {vocab_size}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    print(f"Read checkpoint from {checkpoint_file}")
    model = load_gpt2_model(vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model")
    torch.save(model.state_dict(), os.path.join(output_dir, checkpoint_file))
    print(f"Saved model at {output_dir}")
    with open(os.path.join(output_dir, char_to_idx_file), "w") as f:
        json.dump(char_to_idx, f)
    print(f"Saved {char_to_idx_file} at {output_dir}")


if __name__ == "__main__":
    main(parse_args())
