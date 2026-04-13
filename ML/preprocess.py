import re

def clean_data(text: str) -> str:
    text = re.sub(r"\r\n|\r|\n", " ", text)   # line endings
    text = re.sub(r"\s+", " ", text)           # extra spaces
    text = re.sub(r"<.*?>", " ", text)         # html tags
    return text.strip().lower()