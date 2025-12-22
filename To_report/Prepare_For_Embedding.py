import json
import re
from pathlib import Path

INPUT_FILE = "qa_data_minimized.json"     # your cleaned file
OUTPUT_FILE = "qa_data_ready.json"        # safe formatted file for embedding script

# Required fields
REQUIRED_FIELDS = ["set_type", "question", "answer", "label"]

def sanitize_text(text):
    """Remove non-text artifacts and ensure text is usable by embedding models."""
    if text is None:
        return ""
    text = str(text)

    # Remove trailing partial sentences if they cut mid-word due to truncation
    text = re.sub(r"[A-Za-zığüşöçİĞÜŞÖÇ0-9]+$", "", text)

    # Strip whitespace
    return text.strip()


def validate_and_fix_entry(entry, index):
    """Ensure a single QA entry is structurally valid."""
    fixed = {}

    # Fill missing fields with safe defaults
    for field in REQUIRED_FIELDS:
        if field not in entry or entry[field] is None:
            print(f"[WARN] Entry {index}: Missing field {field}, inserting placeholder.")
            if field == "set_type":
                fixed[field] = "train"
            elif field == "label":
                fixed[field] = -1
            else:
                fixed[field] = ""
        else:
            fixed[field] = entry[field]

    # Clean question/answer text
    fixed["question"] = sanitize_text(fixed["question"])
    fixed["answer"] = sanitize_text(fixed["answer"])

    # Ensure label is int
    try:
        fixed["label"] = int(fixed["label"])
    except:
        fixed["label"] = -1

    return fixed


def main():
    print("=== Preparing dataset for embedding generation ===")

    if not Path(INPUT_FILE).exists():
        print(f"ERROR: Input file '{INPUT_FILE}' not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    seen = set()

    for i, entry in enumerate(data):
        fixed = validate_and_fix_entry(entry, i)

        # Create dedupe key (question + label + set_type)
        key = (fixed["question"], fixed["answer"], fixed["label"], fixed["set_type"])

        if key in seen:
            print(f"[INFO] Duplicate removed at index {i}.")
            continue

        seen.add(key)
        cleaned.append(fixed)

    print(f"Final dataset size: {len(cleaned)} entries")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to '{OUTPUT_FILE}'. This file is ready for embedding.")


if __name__ == "__main__":
    main()
