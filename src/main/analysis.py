import os
from argparse import ArgumentParser, Namespace
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from data.tatoeba import load_tatoeba

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args: Namespace = parser.parse_args()
    df: pd.DataFrame = load_tatoeba(args.data_dir)
    sentences: List[str] = df["sentence"].tolist()
    character_to_count = dict()
    for sentence in sentences:
        for character in sentence:
            character_to_count[character] = character_to_count.get(character, 0) + 1
    print(f"Number of unique characters: {len(character_to_count)}")
    character_to_count = [(character, count) for character, count in character_to_count.items()]
    character_to_count = sorted(character_to_count, key=lambda x: x[1], reverse=True)
    print(f"Most common characters:")
    for character, count in character_to_count:
        print(f"Character: {character}, Count: {count}")
    print()
    print(f"Number of sentences: {len(sentences)}")
    print(f"Number of unique sentences: {len(set(sentences))}")
    length_to_count = dict()
    for sentence in sentences:
        length = len(sentence)
        length_to_count[length] = length_to_count.get(length, 0) + 1
    length_to_count = [(length, count) for length, count in length_to_count.items()]
    num_characters = sum(length * count for length, count in length_to_count)
    print(f"Number of characters: {num_characters}")
    lengths = []
    counts = []
    for length, count in sorted(length_to_count, key=lambda x: x[0], reverse=False):
        lengths.append(length)
        counts.append(count)
    print(f"Min sentence length: {min(lengths)}")
    print(f"Max sentence length: {max(lengths)}")
    lengths = lengths[:200]
    counts = counts[:200]
    print(lengths)
    print(counts)
    plt.figure(figsize=(100, 100))
    plt.bar(lengths, counts, color="skyblue", edgecolor="black")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.title("Sentence Length Distribution")
    plt.xticks(lengths)  # Ensure all unique values are displayed on x-axis
    plt.savefig("value_counts.png")

    print()
    print(f"Most common sentence lengths:")
    for length, count in sorted(length_to_count, key=lambda x: x[1], reverse=True):
        print(f"Length: {length}, Count: {count}")
