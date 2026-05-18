import torch
from torch.utils.data import Dataset


def has_input(text):
    if not isinstance(text, str):
        return False

    cleaned = text.strip().lower()

    invalid_inputs = {
        "",
        "no input",
        "noinput",
        "<noinput>",
        '"<noinput>"',
        "<no input>",
        "no input required",
        "no input required.",
        "n/a",
    }

    return cleaned not in invalid_inputs


def format_input_alpaca(entry):
    if has_input(entry["input"]):
        return f"""Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{entry["instruction"]}

### Input:
{entry["input"]}"""

    else:
        return f"""Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{entry["instruction"]}"""


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.encoded_texts = []

        for _, entry in data.iterrows():
            instruction_input = format_input_alpaca(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            full = instruction_input + response
            input_ids = tokenizer.encode(full)
            prompt_ids = tokenizer.encode(instruction_input + "\n\n### Response:\n")

            self.encoded_texts.append(
                {"input_ids": input_ids, "prompt_length": len(prompt_ids)}
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device: torch.device | str = "cpu",
):
    """
    Custom collate function to prepare batches to train GPT-2.
    Adds padding for batch length consistency.
    Creates inputs and targets shifted by one position.
    Masks prompt tokens and extra padding tokens from the loss.
    """

    processed_batch = []

    for item in batch:
        input_ids = item["input_ids"].copy()
        prompt_length = item["prompt_length"]

        # Truncate before padding
        if allowed_max_length is not None:
            input_ids = input_ids[:allowed_max_length]
            prompt_length = min(prompt_length, allowed_max_length)

        processed_batch.append(
            {
                "input_ids": input_ids,
                "prompt_length": prompt_length,
            }
        )

    batch_max_length = max(len(item["input_ids"]) + 1 for item in processed_batch)

    inputs_list, targets_list = [], []

    for item in processed_batch:
        input_ids = item["input_ids"].copy()
        prompt_length = item["prompt_length"]

        # Add EOS token
        input_ids += [pad_token_id]

        # Pad to batch max length
        padded = input_ids + [pad_token_id] * (batch_max_length - len(input_ids))

        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        # Mask prompt/instruction tokens
        targets[: max(prompt_length - 1, 0)] = ignore_index

        # Mask extra padding tokens, but keep first EOS as valid target
        mask = targets == pad_token_id
        indices = torch.nonzero(mask, as_tuple=True)[0]

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list)
    targets_tensor = torch.stack(targets_list)

    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    import pandas as pd
    import tiktoken
    from torch.utils.data import DataLoader
    from Train.load_pretrained_gpt2 import pretrained_gpt2_generator
    from Train.train import train, calc_loss_batch, calc_loss_loader
    from Models.config import GPT_CONFIG_124M
    from functools import partial

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    df = pd.read_parquet(
        "hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
    )
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_portion = int(len(df) * 0.85)
    test_portion = int(len(df) * 0.1)
    val_portion = len(df) - train_portion - test_portion

    train_data = df.iloc[:train_portion]
    test_data = df.iloc[train_portion : train_portion + test_portion]
    val_data = df.iloc[train_portion + test_portion :]

    tokenizer = tiktoken.get_encoding("gpt2")
    num_workers = 0
    batch_size = 8

    train_dataset = InstructionDataset(train_data, tokenizer)
    customized_collate_fn = partial(
        custom_collate, allowed_max_length=256, device=device
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    model = pretrained_gpt2_generator(GPT_CONFIG_124M)

    model.to(device)

    lr = 5e-5
    weight_decay = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 1

    start_context = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Explain what machine learning is

### Response:
"""

    train_losses, val_losses, tokens_seen = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=epochs,
        eval_freq=100,
        eval_iter=1,
        start_context=start_context,
        tokenizer=tokenizer,
        batch_loss_fn=calc_loss_batch,
        loader_loss_fn=calc_loss_loader,
        save_weights=True,
        train_name="instruction",
    )
