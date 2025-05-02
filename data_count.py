import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import datasets
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4_local", type=str_to_bool, default=True)
    parser.add_argument("--train_data_path", type=str, default="en/c4-train.*.json.gz",
                        help="Path to C4 training data files")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for quick testing)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup multi-gpu if available
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        device = "cuda:0"
    else:
        world_size = 1
        device = "cpu"

    print(f"Using device: {device}, world_size: {world_size}")
    gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
    print(f"Gradient accumulation steps: {gradient_accumulation}")

    # Load dataset
    start_time = time.time()
    if args.c4_local:
        print(f"Loading local C4 dataset from: {args.train_data_path}")
        data = datasets.load_dataset('arrow', data_files=args.train_data_path, split="train", streaming=True)
    else:
        print("Loading C4 dataset from HuggingFace")
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base",
        model_max_length=args.max_length,
    )

    # Shuffle data
    seed_for_shuffle = args.seed
    print(f"Shuffling data with seed {seed_for_shuffle}")
    data = data.shuffle(seed=seed_for_shuffle)

    # Define preprocessing function
    def preprocess_batched(batch):
        tokenized = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return tokenized

    # Sample counting variables
    total_samples = 0
    total_tokens = 0
    total_text_chars = 0
    batches_processed = 0

    # Set up for token counting
    pad_idx = tokenizer.pad_token_id

    # Time tracking
    process_start_time = time.time()

    # Process the dataset
    print("Starting to process dataset...")
    pbar = tqdm(desc="Processing samples")

    batch_texts = []

    for sample in data:
        batch_texts.append(sample["text"])
        total_samples += 1
        total_text_chars += len(sample["text"])

        if len(batch_texts) >= args.batch_size:
            # Process batch
            batch = preprocess_batched({"text": batch_texts})
            tokens_in_batch = (batch["input_ids"] != pad_idx).sum().item()
            total_tokens += tokens_in_batch

            batches_processed += 1
            batch_texts = []

            # Update progress
            if batches_processed % 100 == 0:
                elapsed = time.time() - process_start_time
                avg_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0
                samples_per_second = total_samples / elapsed if elapsed > 0 else 0
                tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0

                pbar.update(100)
                pbar.set_postfix({
                    'samples': total_samples,
                    'tokens': total_tokens,
                    'tok/sample': f'{avg_tokens_per_sample:.1f}',
                    'samples/s': f'{samples_per_second:.1f}',
                    'tokens/s': f'{tokens_per_second:.1f}'
                })

            # Check if we've reached the maximum number of samples to process
            if args.max_samples is not None and total_samples >= args.max_samples:
                break

    # Process any remaining samples
    if batch_texts:
        batch = preprocess_batched({"text": batch_texts})
        tokens_in_batch = (batch["input_ids"] != pad_idx).sum().item()
        total_tokens += tokens_in_batch

    total_time = time.time() - process_start_time

    # Calculate statistics
    avg_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0
    avg_chars_per_sample = total_text_chars / total_samples if total_samples > 0 else 0
    samples_per_second = total_samples / total_time if total_time > 0 else 0
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    batches_per_second = batches_processed / total_time if total_time > 0 else 0

    # Calculate training steps and epochs
    update_steps_per_epoch = total_samples / args.total_batch_size

    print("\n" + "=" * 50)
    print("DATASET STATISTICS:")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total text characters: {total_text_chars:,}")
    print(f"Average tokens per sample: {avg_tokens_per_sample:.2f}")
    print(f"Average characters per sample: {avg_chars_per_sample:.2f}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Processing rate: {samples_per_second:.2f} samples/second")
    print(f"Token processing rate: {tokens_per_second:.2f} tokens/second")
    print(f"Batch processing rate: {batches_per_second:.2f} batches/second")
    print("\n" + "=" * 50)
    print("TRAINING ESTIMATES:")
    print(f"Batch size: {args.batch_size}")
    print(f"Total batch size (across GPUs): {args.total_batch_size}")
    print(f"Update steps per epoch: {update_steps_per_epoch:.2f}")

    # Print estimates for different training durations
    for num_steps in [1000, 5000, 10000, 50000, 100000]:
        tokens_processed = num_steps * args.total_batch_size * avg_tokens_per_sample
        tokens_in_billions = tokens_processed / 1_000_000_000
        print(f"For {num_steps} update steps: ~{tokens_in_billions:.2f}B tokens processed")

    print("=" * 50)


if __name__ == "__main__":
    main()