import os
from argparse import ArgumentParser, Namespace
from data.util import augment_sentences


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Augment sentences.")
    parser.add_argument(
        "--sentences-path",
        type=str,
        help="Path to the sentences file.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to the input prefixes.",
    )
    parser.add_argument(
        "--answer-path",
        type=str,
        help="Path to the answers.",
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
    args = parser.parse_args()
    if not args.sentences_path and not (args.input_path and args.answer_path):
        raise ValueError(
            "Either --sentences-path or (--input-path and --answer-path) is required"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.sentences_path:
        sentences_path = args.sentences_path
    else:
        with open(args.input_path, "r") as f:
            sentences = f.readlines()
            sentences = [sentence[:-1] for sentence in sentences]
        with open(args.answer_path, "r") as f:
            answers = f.readlines()
            if len(answers) != len(sentences):
                raise ValueError(
                    "inputs and answers have different lengths: "
                    f"{len(sentences)} vs {len(answers)}"
                )
            for i, answer in enumerate(answers):
                sentences[i] = sentences[i] + answer[0]
        sentences_path = os.path.join(args.output_dir, "sentences.txt")
        with open(sentences_path, "w") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
    augment_sentences(sentences_path, args.output_dir, min_len=args.min_len)


if __name__ == "__main__":
    main()
