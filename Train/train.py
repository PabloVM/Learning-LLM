import torch
import torch_directml
from Models.gpt2 import GPTModel
from Models.config import GPT_CONFIG_124M_LOW as GPT_CONFIG_124M

from Data.dataset_creation import create_dataloader
import tiktoken

from pathlib import Path


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=50):
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Keep only the last context_size tokens because the model cannot process sequences longer than its context window
        idx_cond = idx[:, -context_size:]

        # Disable gradient computation during inference
        with torch.no_grad():
            # Forward pass through the model logits shape: (batch_size, seq_len, vocab_size)
            logits = model(idx_cond)

        # Select logits from the last generated token We only care about predicting the next token Shape:(batch_size, vocab_size)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # Convert logits into probabilities
        probs = torch.softmax(logits, dim=-1)

        # Select the token with highest probability Shape: (batch_size, 1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append the predicted token to the sequence Previous: (batch_size, seq_len) After concatenation:(batch_size, seq_len + 1)
        idx = torch.cat((idx, idx_next), dim=-1)

    # Return complete generated sequence
    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(
            0, 1
        ),  # logits (batch_size, seq_len, voc_size) -> (predictions, voc_size)
        target_batch.flatten(),  # Flatten target tokens into a single vector
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return torch.nan
    elif num_batches is None:
        num_batches = len(
            data_loader
        )  # If no number of batches specified, iterates over all
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(
    model, train_loader, val_loader, device, eval_iter, loss_fn=calc_loss_loader
):
    model.eval()  # Puts the model in eval mode, disables dropouts
    with torch.no_grad():  # Disables gradient tracking
        train_loss = loss_fn(train_loader, model, device, num_batches=eval_iter)
        validation_loss = loss_fn(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, validation_loss


def generate_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=0.8,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train(
    model,
    train_data_loader,
    val_data_loader,
    optimizer,
    device,
    epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    loader_loss_fn=calc_loss_loader,
    batch_loss_fn=calc_loss_batch,
    save_weights=False,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(epochs):
        model.train()  # Puts the model in train mode
        for input_batch, target_batch in train_data_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch train
            loss = batch_loss_fn(input_batch, target_batch, model, device)

            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights

            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_data_loader,
                    val_data_loader,
                    device,
                    eval_iter,
                    loader_loss_fn,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch:{epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f} Validation loss {val_loss:.3f}"
                )
        if save_weights:
            weights_dir = Path(__file__).parent / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                weights_dir / f"{model.__class__.__name__}_{epoch}.pt2",
            )

        generate_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / "Data" / "dracula.txt"
    tokenizer = tiktoken.get_encoding("gpt2")

    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(raw_text))

    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    train_loader = create_dataloader(
        train_text,
        batch_size=1,
        max_length=64,
        stride=64,
    )

    val_loader = create_dataloader(
        val_text,
        batch_size=1,
        max_length=64,
        stride=64,
    )
    torch.manual_seed(123)

    lr = 4e-4
    weight_decay = 0.1

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        try:
            device = torch_directml.device()
        except Exception:
            device = torch.device("cpu")

    print(f"Using device: {device}")
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 10

    train_losses, val_losses, tokens_seen = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        batch_loss_fn=calc_loss_batch,
        loader_loss_fn=calc_loss_loader,
    )
