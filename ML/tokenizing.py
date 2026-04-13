from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 128

def tokenizing_raw_data(row) -> dict:
    input_text = "summarize: " + row["dialogue"]

    inputs = tokenizer(
        input_text,
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