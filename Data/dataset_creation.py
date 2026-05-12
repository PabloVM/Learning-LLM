import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path


file_path = Path(__file__).parent / "dracula.txt"

with open(file_path, "r", encoding="utf-8") as file:
    raw_text = file.read()


class GPTDataset(Dataset):
    def __init__(self, txt: str, tokenizer, max_lenght: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_lenght, stride):
            input_batch = token_ids[i : i + max_lenght]
            output_batch = token_ids[i + 1 : i + 1 + max_lenght]

            self.input_ids.append(torch.tensor(input_batch))
            self.target_ids.append(torch.tensor(output_batch))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


dataloader = create_dataloader(
    raw_text, batch_size=1, max_length=4, stride=3, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
first_batch = next(data_iter)
print(first_batch)
