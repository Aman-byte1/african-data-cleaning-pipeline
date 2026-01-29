from datasets import load_dataset

SOURCE_REPO = "amanuelbyte/finetranslations-sentence-level"
LANG_CONFIG = "afr_Latn"

try:
    dataset = load_dataset(SOURCE_REPO, LANG_CONFIG, split="train", streaming=True)
    first_item = next(iter(dataset))
    print("Dataset keys:", first_item.keys())
    print("Sample item:", first_item)
except Exception as e:
    print(f"Error: {e}")
