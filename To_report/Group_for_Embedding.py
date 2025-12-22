import json

INPUT = "qa_data_ready.json"
OUTPUT = "qa_data_grouped.json"

print(f"Loading {INPUT}...")
with open(INPUT, "r", encoding="utf-8") as f:
    flat = json.load(f)

train = []
test = []

for entry in flat:
    if entry.get("set_type") == "train":
        train.append(entry)
    elif entry.get("set_type") == "test":
        test.append(entry)

print(f"Train size: {len(train)}")
print(f"Test size:  {len(test)}")

grouped = {
    "train": train,
    "test": test
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(grouped, f, indent=2, ensure_ascii=False)

print(f"\nGrouped dataset saved to {OUTPUT}")
