from pathlib import Path
import re

file_path = Path(__file__).parent / "dracula.txt"

with open(file_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

## Raw text pre processing

tokens = re.split(
    r'([,.:;?_!"\[\]()\']|--|\s)', raw_text
)  # Split text based on common punctuation and spaces
all_words = sorted(
    set([item for item in tokens if item.strip()])
)  # Create the unique vocabulary for the given corpus
all_words.extend(["<|unk|>", "<|endoftext|>"])  # Extended vocabulary


class SimpleTokenizer:
    def __init__(self, vocab: list):
        self.vocab = {
            token: integer for integer, token in enumerate(vocab)
        }  # Dictionary containing tokens and ids
        self.inverse_vocab = {
            v: k for k, v in self.vocab.items()
        }  # Reverse dictionary for decoding

    def encode(self, text: str):
        preprocessed = re.split(r'([,.:;?_!"\[\]()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item.strip() if item in self.vocab else "<|unk|>" for item in preprocessed
        ]
        ids = [self.vocab[token] for token in preprocessed]
        return ids

    def decode(self, ids: list):
        text = " ".join([self.inverse_vocab[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


tokenizer = SimpleTokenizer(all_words)

sample_text = "My dog is already here dsfds"
encoded_text = tokenizer.encode(sample_text)
print(encoded_text)
decoded_text = tokenizer.decode(encoded_text)
print(decoded_text)
