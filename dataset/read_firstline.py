import os
import regex # Use the enhanced regex library for Unicode properties
import random
from tqdm.auto import tqdm # For progress bars with .map()
from datasets import load_dataset, DatasetDict, Dataset
from typing import List
from pathlib import Path

# --- Assume your setup code is here ---
# from dataset import find_files, directories
# Dummy implementation for example:
def find_files(dirs):
    # Replace with your actual find_files logic
    # Example: return ['path/to/file1.parquet', 'path/to/file2.parquet']
    print("Warning: Using dummy find_files. Replace with your actual implementation.")
    # Create dummy files for demonstration if they don't exist
    dummy_dir = Path("./dummy_parquet_data")
    dummy_dir.mkdir(exist_ok=True)
    dummy_files = [dummy_dir / "data1.parquet", dummy_dir / "data2.parquet"]
    if not all(f.exists() for f in dummy_files):
        print("Creating dummy parquet files...")
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        data1 = pd.DataFrame({'text': ["Hello world", "你好，世界！", " মিশ্রিত টেক্সট mixed text"]})
        data2 = pd.DataFrame({'text': ["这是一段纯中文。", "Another English sentence.", "こんにちは世界"]})
        pq.write_table(pa.Table.from_pandas(data1), dummy_files[0])
        pq.write_table(pa.Table.from_pandas(data2), dummy_files[1])
    return [str(f) for f in dummy_files]

directories = ["./dummy_parquet_data"] # Replace with your actual directories
# --- End Setup ---


# Configuration for sampling
SAMPLE_SIZE = 10000  # Number of examples to check (adjust as needed)
SEED = 42          # For reproducible shuffling

# --- Find and load dataset ---
data_files: List[str] = find_files(directories)
if not data_files:
    raise FileNotFoundError("No parquet files found in the specified directories.")

cache_dir = "./data/huggingface_cache/datasets"
os.makedirs(cache_dir, exist_ok=True)

try:
    print("Loading dataset...")
    # Load the dataset (only the 'text' column)
    dataset: DatasetDict = load_dataset(
        "parquet",
        data_files=data_files,
        columns=["text"],
        cache_dir=cache_dir
    )
    print("Dataset loaded.")

    # --- Check Chinese Character Proportion ---

    if not dataset:
        print("The loaded dataset dictionary is empty.")
    else:
        # Get the first split (usually 'train' if not specified)
        first_split_name = list(dataset.keys())[0]
        print(f"\nAnalyzing split: '{first_split_name}'")
        split_dataset: Dataset = dataset[first_split_name]

        if len(split_dataset) == 0:
            print(f"The split '{first_split_name}' is empty.")
        else:
            print(f"Total examples in split: {len(split_dataset)}")

            # --- Sampling ---
            num_samples = min(SAMPLE_SIZE, len(split_dataset))
            print(f"Sampling {num_samples} examples for analysis (seed={SEED})...")
            # Shuffle and select a subset for efficiency
            sampled_dataset = split_dataset.shuffle(seed=SEED).select(range(num_samples))

            # --- Define function to count characters ---
            # Use \p{Han} which matches CJK unified ideographs (most Chinese chars)
            # You might expand this if needed (e.g., include CJK Compatibility Ideographs)
            chinese_char_pattern = regex.compile(r'\p{Han}')

            def count_chinese_chars(example):
                """Counts Chinese and total characters in a text example."""
                text = example.get("text", "") # Get text, default to empty string if missing
                if not isinstance(text, str):
                    text = str(text) # Attempt to convert to string if not already

                # Find all matches using the precompiled regex pattern
                chinese_chars = chinese_char_pattern.findall(text)
                # Count total non-whitespace characters for a potentially more meaningful ratio
                # Or use len(text) for absolute total characters including spaces/punctuation
                total_chars = len(text)
                # total_chars_no_space = len(regex.sub(r'\s+', '', text)) # Alternative: ignore whitespace

                return {
                    "chinese_char_count": len(chinese_chars),
                    "total_char_count": total_chars
                }

            # --- Apply the counting function using .map() ---
            print("Calculating character counts on the sample (this might take a moment)...")
            # batched=True is generally faster
            # num_proc can be set for multiprocessing if dataset is large and machine has cores
            counted_dataset = sampled_dataset.map(
                count_chinese_chars,
                batched=False, # Process example by example for simplicity here
                # batched=True, # Usually faster for large datasets
                # batch_size=1000, # Adjust batch size if using batched=True
                # num_proc=os.cpu_count() // 2 # Optional: use multiple cores
                desc="Counting Characters" # Progress bar description
            )
            print("Calculation complete.")

            # --- Aggregate the counts ---
            total_chinese_chars = sum(counted_dataset["chinese_char_count"])
            total_chars_in_sample = sum(counted_dataset["total_char_count"])

            # --- Calculate and print the proportion ---
            if total_chars_in_sample > 0:
                proportion = (total_chinese_chars / total_chars_in_sample) * 100
                print("\n--- Analysis Results (Based on Sample) ---")
                print(f"Total characters analyzed in sample: {total_chars_in_sample:,}")
                print(f"Total Chinese characters found in sample: {total_chinese_chars:,}")
                print(f"Estimated proportion of Chinese characters: {proportion:.2f}%")
                print("-------------------------------------------")
            elif len(sampled_dataset) > 0:
                 print("\n--- Analysis Results (Based on Sample) ---")
                 print("The sampled text contains 0 characters.")
                 print("-------------------------------------------")
            else:
                 print("\n--- Analysis Results ---")
                 print("Sampled dataset is empty.")
                 print("-------------------------------------------")


except FileNotFoundError as e:
    print(f"Error: File not found. Details: {e}")
except ImportError as e:
     print(f"Error: Missing library. Please install it. Details: {e}")
     print("Try: pip install regex datasets pyarrow pandas tqdm")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging