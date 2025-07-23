import os
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import time
from loguru import logger


def download_and_prepare_c4():
    logger.info("Starting C4 dataset download and preparation...")
    start_time = time.time()

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    logger.info("Downloading training split...")
    train_dataset = datasets.load_dataset("allenai/c4", "en", split="train", streaming=False)

    logger.info("Downloading validation split...")
    val_dataset = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=False)

    def check_dataset_size():

        try:
            train_size = len(train_dataset)
            val_size = len(val_dataset)
            logger.info(f"Training set size: {train_size:,} examples")
            logger.info(f"Validation set size: {val_size:,} examples")
            return True
        except Exception as e:
            logger.error(f"Error checking dataset size: {e}")
            return False

    logger.info("Testing dataset and tokenizer...")
    try:
        sample_text = train_dataset[0]['text']
        tokens = tokenizer(sample_text, truncation=True, max_length=256)
        logger.info("Successfully tested tokenization on sample data")
    except Exception as e:
        logger.error(f"Error testing tokenization: {e}")
        return False

    if not check_dataset_size():
        logger.error("Failed to verify dataset size")
        return False

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Dataset preparation completed in {duration:.2f} seconds")

    cache_dir = os.getenv("HF_HOME")
    if cache_dir:
        dataset_path = os.path.join(cache_dir, "datasets")
        if os.path.exists(dataset_path):
            cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, _, filenames in os.walk(dataset_path)
                             for filename in filenames)
            logger.info(f"Total cache size: {cache_size / (1024 ** 3):.2f} GB")

    logger.info("C4 dataset is ready for use!")
    return True


if __name__ == "__main__":
    success = download_and_prepare_c4()
    if success:
        logger.info("Dataset preparation completed successfully!")
    else:
        logger.error("Dataset preparation failed!")