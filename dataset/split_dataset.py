import json
import os
import random


def split_dataset(input_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 1. Validation: Ensure ratios sum to 1
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    print(f"Loading data from {input_file}...")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

    total_count = len(data)
    print(f"Total items found: {total_count}")

    # 2. Shuffle data for a random distribution (Critical for ML)
    # Setting a seed ensures the split is reproducible
    random.seed(42)
    random.shuffle(data)

    # 3. Calculate Split Indices
    # We use explicit integer slicing to avoid "off-by-one" errors
    train_end = int(total_count * train_ratio)
    val_end = int(total_count * (train_ratio + val_ratio))

    # 4. Slice the Data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 5. Define Output Filenames
    base_name = os.path.splitext(input_file)[0]
    # Adjusting filenames to match your exact request format
    # Assuming input is "_dataset.json", base is "_dataset"
    # Logic handles generic names, but hardcoding your specific request below

    outputs = [
        ("./dataset/_train_dataset.json", train_data),
        ("./dataset/_validation_dataset.json", val_data),
        ("./dataset/_test_dataset.json", test_data),
    ]

    # 6. Save Files
    for filename, subset_data in outputs:
        print(f"Saving {len(subset_data)} items to {filename}...")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(subset_data, f, indent=4)

    print("✅ Split complete.")


if __name__ == "__main__":
    import math  # Imported here to keep the function clean if pasted elsewhere

    # Configuration
    INPUT_FILENAME = "/kaggle/input/autism-jdc/_dataset.json"

    split_dataset(INPUT_FILENAME, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
