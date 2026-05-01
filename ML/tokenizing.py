from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

MAX_INPUT_LENGTH  = 512    # Reduced from 1024 to speed up training
MAX_TARGET_LENGTH = 128

def tokenizing_raw_data(row) -> dict:
    # ⚠️ No "summarize: " prefix — BART doesn't use task prefixes
    inputs = tokenizer(
        row["dialogue"],
        padding="max_length",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    targets = tokenizer(
        row["summary"],
        padding="max_length",
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    inputs["labels"] = [
        token_id if token_id != tokenizer.pad_token_id else -100
        for token_id in targets["input_ids"]
    ]

    return inputs