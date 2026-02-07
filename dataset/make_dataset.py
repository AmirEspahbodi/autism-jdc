import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    """Safe loading of JSON data with error handling."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {filepath}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"File {filepath} contains invalid JSON.")


def save_json_data(filepath: str, data: Dict[str, Any]) -> None:
    """Saves a dictionary to a JSON file with pretty printing."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"✅ Successfully saved match to: {filepath}")
    except IOError as e:
        print(f"❌ Error writing to file {filepath}: {e}")


def find_matching_item(
    ds1: List[Dict[str, Any]], ds2: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Finds the item in ds1 where item['id'] matches any item['key'] in ds2.
    Optimized to O(N + M) time complexity using a Set.
    """
    new_dataset = []
    for item in ds1:
        id, input_prompt = item["id"], item["prompt"]

        for item2 in ds2:
            if id == item2["key"]:
                model_output = item2["value"]

        new_item = {
            "id": id,
            "input_prompt": input_prompt,
            "model_output": model_output,
        }
        new_dataset.append(new_item)

    return new_dataset


def main():
    # 1. Load the data
    dataset1 = load_json_data("./dataset/_1prompts.json")
    dataset2 = load_json_data("./dataset/_2initial_prompts_outputs.json")

    print(f"Loaded {len(dataset1)} items from Dataset 1.")
    print(f"Loaded {len(dataset2)} items from Dataset 2.")

    # 2. Find the match
    new_dataset = find_matching_item(dataset1, dataset2)

    # 3. Save the result
    if new_dataset:
        save_json_data("./dataset/_dataset.json", new_dataset)
    else:
        print("\n⚠️ No matching item found. No file was saved.")


if __name__ == "__main__":
    main()
