import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    """
    Dataset class that specifies how data is loaded and process.
    This class ensures data is properly loaded and all examples have same length by apdding it with padding_token
    """

    def __init__(self, dataframe, tokenizer, max_lenght=None, pad_token_id=50256):
        self.data = dataframe
        # Tokenize each sentence
        self.encoded_texts = [
            tokenizer.encode(sentence) for sentence in self.data["sentence"]
        ]

        if max_lenght is None:
            # If no max_lenght is specified a custom function is used to get the largest encoded sentence
            self.max_lenght = self._longest_encoded_length()
        else:
            self.max_lenght = max_lenght
            # If max_length is specified, samples are trimmed to match.
            self.encoded_texts = [
                encoded_text[: self.max_lenght] for encoded_text in self.encoded_texts
            ]
        # Apply padding to encoded texts to ensure same length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_lenght - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_lenght = 0
        for encoded_text in self.encoded_texts:
            encoded_lenght = len(encoded_text)
            if encoded_lenght > max_lenght:
                max_lenght = encoded_lenght
        return max_lenght


def calc_accuracy(data_loader, model, device, num_batches=None):
    model.eval()
    correct, examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
                predicted = torch.argmax(logits, dim=-1)
                examples += predicted.shape[0]
                correct += (predicted == target_batch).sum().item()
            print(f"Accuracy batch {i} : {correct / examples}")
        else:
            break

    return correct / examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss(data_loader, model, device, num_batches=None):
    total_loss = 0

    if len(data_loader) == 0:
        return torch.nan
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break

    return total_loss


if __name__ == "__main__":
    import pandas as pd
    import tiktoken
    from torch.utils.data import DataLoader
    from Train.load_pretrained_gpt2 import pretrained_gpt2_generator
    from Train.train import train
    from Models.config import GPT_CONFIG_124M

    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    train_df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["train"])
    validation_df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["train"])

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SentimentDataset(dataframe=train_df, tokenizer=tokenizer)
    validation_dataset = SentimentDataset(dataframe=validation_df, tokenizer=tokenizer)
    test_dataset = SentimentDataset(dataframe=test_df, tokenizer=tokenizer)

    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = pretrained_gpt2_generator(GPT_CONFIG_124M)

    for params in model.parameters():
        params.requires_grad = False

    model.out_head = torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"], out_features=2
    )

    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.last_norm.parameters():
        param.requires_grad = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    lr = 4e-4
    weight_decay = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses, examples_seen = train(
        model,
        train_loader,
        validation_loader,
        optimizer,
        device,
        5,
        50,
        5,
        "This film is absolute trash",
        tokenizer,
        batch_loss_fn=calc_loss_batch,
        loader_loss_fn=calc_loss,
    )
