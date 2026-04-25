import argparse
import os

from datasets import load_dataset, load_from_disk


def build_cache_path(base_dir, dataset, split, seed, num_samples):
    dataset_safe = dataset.replace("/", "_")
    return os.path.join(base_dir, f"{dataset_safe}_{split}_n{num_samples}_seed{seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Huggingface dataset name"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hf_cache_dir", type=str, default="./hf_cache")
    parser.add_argument("--sample_cache_dir", type=str, default="./dataset_samples")

    args = parser.parse_args()

    sample_cache_path = build_cache_path(
        args.sample_cache_dir, args.dataset, args.split, args.seed, args.num_samples
    )

    if os.path.exists(sample_cache_path):
        print(f"Lade vorhandenen Sample-Cache: {sample_cache_path}")
        dataset = load_from_disk(sample_cache_path)
    else:
        print("Lade Dataset von Hugging Face...")
        dataset = load_dataset(
            args.dataset, split=args.split, cache_dir=args.hf_cache_dir
        )

        print("Erzeuge reproduzierbare Zufallsstichprobe...")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

        os.makedirs(sample_cache_path, exist_ok=True)
        print(f"Speichere Sample-Cache: {sample_cache_path}")
        dataset.save_to_disk(sample_cache_path)

    print(dataset)
    for data in dataset:
        print(data)


if __name__ == "__main__":
    main()
