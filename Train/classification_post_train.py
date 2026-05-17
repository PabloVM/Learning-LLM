import torch
from torch.utils.data import Dataset
from pathlib import Path


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


def get_last_real_token_logits(input_batch, model, pad_token_id=50256):
    logits = model(input_batch)

    # Compare each token with padding
    attention_mask = input_batch != pad_token_id
    # Sum all True values in mask to get last true token position
    last_token_idx = attention_mask.sum(dim=1) - 1
    # Get batch index
    batch_idx = torch.arange(input_batch.size(0), device=input_batch.device)

    return logits[batch_idx, last_token_idx]


def sentiment_loss_batch(input_batch, target_batch, model, device, pad_token_id=50256):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = get_last_real_token_logits(input_batch, model, pad_token_id)

    return torch.nn.functional.cross_entropy(logits, target_batch)


def evaluate_sentiment(
    model, data_loader, device, num_batches=None, pad_token_id=50256
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_examples = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits = get_last_real_token_logits(input_batch, model, pad_token_id)
            loss = torch.nn.functional.cross_entropy(logits, target_batch)

            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            correct += (preds == target_batch).sum().item()
            total_examples += target_batch.size(0)

    avg_loss = total_loss / num_batches
    accuracy = correct / total_examples

    model.train()

    return avg_loss, accuracy


def train_sentiment(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=5,
    eval_freq=50,
    eval_iter=None,
    pad_token_id=50256,
    save_weights=True,
    weights_dir="weights",
):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    examples_seen = []
    global_step = -1
    seen_examples = 0

    for epoch in range(epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = sentiment_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
                pad_token_id=pad_token_id,
            )

            loss.backward()
            optimizer.step()

            seen_examples += input_batch.size(0)
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, train_acc = evaluate_sentiment(
                    model=model,
                    data_loader=train_loader,
                    device=device,
                    num_batches=eval_iter,
                    pad_token_id=pad_token_id,
                )

                val_loss, val_acc = evaluate_sentiment(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    num_batches=eval_iter,
                    pad_token_id=pad_token_id,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                examples_seen.append(seen_examples)

                print(
                    f"Epoch:{epoch + 1} "
                    f"(Step {global_step:06d}) | "
                    f"Train loss {train_loss:.3f}, acc {train_acc:.3f} | "
                    f"Val loss {val_loss:.3f}, acc {val_acc:.3f}"
                )

        if save_weights:
            weights_dir = Path(weights_dir)
            weights_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                model.state_dict(),
                weights_dir / f"{model.__class__.__name__}_epoch_{epoch + 1}.pt",
            )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "examples_seen": examples_seen,
    }


if __name__ == "__main__":
    import pandas as pd
    import tiktoken
    from torch.utils.data import DataLoader
    import torch_directml
    from Train.load_pretrained_gpt2 import pretrained_gpt2_generator
    from Models.config import GPT_CONFIG_124M

    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
    }
    train_df = pd.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["train"])
    validation_df = pd.read_parquet(
        "hf://datasets/stanfordnlp/sst2/" + splits["validation"]
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SentimentDataset(dataframe=train_df, tokenizer=tokenizer)
    validation_dataset = SentimentDataset(dataframe=validation_df, tokenizer=tokenizer)

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
        try:
            device = torch_directml.device()
        except Exception:
            device = torch.device("cpu")
    print(f"Using devide: {device}")

    model.to(device)

    lr = 1e-4
    weight_decay = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train = train_sentiment(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        device=device,
        epochs=5,
        eval_freq=50,
        eval_iter=20,
        pad_token_id=50256,
        save_weights=True,
    )
