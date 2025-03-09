from torch.utils.data import DataLoader

from data.dist import sequences_to_batches, LengthGroupedBatchDataset


def test_sequences_to_batches():
    sequences = [
        [3, 5, 8, 9],
        [2, 7],
        [4, 6, 1],
        [8, 5],
        [7, 3, 9, 2, 6],
        [1, 3, 5],
        [2, 8, 6, 4],
        [9, 1],
        [5, 7],
        [6, 3, 1, 8],
        [7, 8, 3, 5, 2],
        [4, 9, 1],
        [6, 8, 4, 7],
        [2, 5, 3],
        [9, 6],
    ]

    batch_size = 3
    world_size = 2

    full_bundle, non_full_bundle = sequences_to_batches(
        sequences, batch_size, world_size
    )

    full_dataset = LengthGroupedBatchDataset(full_bundle)
    non_full_dataset = LengthGroupedBatchDataset(non_full_bundle)

    full_dataloader = DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=LengthGroupedBatchDataset.collate,
    )
    non_full_dataloader = DataLoader(
        non_full_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=LengthGroupedBatchDataset.collate,
    )

    print("=== Full Bundle ===")
    for batch in full_dataloader:
        print(batch, "\n")

    print("=== Non-Full Bundle ===")
    for batch in non_full_dataloader:
        print(batch, "\n")


def main():
    test_sequences_to_batches()


if __name__ == "__main__":
    main()
