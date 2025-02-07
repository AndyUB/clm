#!/usr/bin/env python
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from main.gpt2_tatoeba import train_gpt2_on_tatoeba
from main.util import gpt2_inference


def parse_args() -> Namespace:
    """
    Parse command-line arguments.
    Supports train and test modes.
    --work_dir specifies where to save the model.
    --test_data specifies the path to the test data.
    --test_output specifies the path to write the test predictions.

    Return:
        The parsed arguments.
    """

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument(
        "--test_data", help="path to test data", default="example/input.txt"
    )
    parser.add_argument(
        "--test_output", help="path to write test predictions", default="pred.txt"
    )
    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
        train_gpt2_on_tatoeba(
            data_path="data/tatoeba", output_path=args.work_dir, k=3
        )
    elif args.mode == "test":
        gpt2_inference(
            model_path=args.work_dir,
            input_path=args.test_data,
            output_path=args.test_output,
            k=3,
        )
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))


if __name__ == "__main__":
    main(parse_args())
