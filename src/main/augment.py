from argparse import ArgumentParser, Namespace
from data.util import augment_sentences


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Augment sentences.")
    parser.add_argument(
        "--sentences-path",
        type=str,
        required=True,
        help="Path to the sentences file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the augmented inputs and answers.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Minimum length of the prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augment_sentences(args.sentences_path, args.output_dir, min_len=args.min_len)


if __name__ == "__main__":
    main()
