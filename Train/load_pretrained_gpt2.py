from transformers import GPT2LMHeadModel
from Models.gpt2 import GPTModel
from Models.config import GPT_CONFIG_124M

"""
The goal of this script is to load pretrained gpt2 weights from HugginFace and load into our implementation
"""


def build_weight_map(n_layers):
    """
    Utility function to map official implementation names with custom implementation names.
    QKV are ommited due to official implementation merge all values in one block while in custom implementation is spread.
    """
    weight_map = {
        "transformer.wte.weight": "tok_emb.weight",
        "transformer.wpe.weight": "pos_emb.weight",
        "transformer.ln_f.weight": "last_norm.scale",
        "transformer.ln_f.bias": "last_norm.shift",
        "lm_head.weight": "out_head.weight",
    }

    for i in range(n_layers):
        hf = f"transformer.h.{i}"
        model = f"transformer_blocks.{i}"

        weight_map[f"{hf}.ln_1.weight"] = f"{model}.norm1.scale"
        weight_map[f"{hf}.ln_1.bias"] = f"{model}.norm1.shift"

        weight_map[f"{hf}.attn.c_proj.weight"] = f"{model}.att.projection.weight"
        weight_map[f"{hf}.attn.c_proj.bias"] = f"{model}.att.projection.bias"

        weight_map[f"{hf}.ln_2.weight"] = f"{model}.norm2.scale"
        weight_map[f"{hf}.ln_2.bias"] = f"{model}.norm2.shift"

        weight_map[f"{hf}.mlp.c_fc.weight"] = f"{model}.feedforward.layers.0.weight"
        weight_map[f"{hf}.mlp.c_fc.bias"] = f"{model}.feedforward.layers.0.bias"

        weight_map[f"{hf}.mlp.c_proj.weight"] = f"{model}.feedforward.layers.2.weight"
        weight_map[f"{hf}.mlp.c_proj.bias"] = f"{model}.feedforward.layers.2.bias"

    return weight_map


weight_map = build_weight_map(GPT_CONFIG_124M["n_layers"])


def validate_weight_map(hf_state, model_state, weight_map):
    """
    Utility function to validate tensor shape missmatching
    """
    for hf_key, model_key in weight_map.items():
        hf_tensor = hf_state[hf_key]
        model_tensor = model_state[model_key]

        if hf_tensor.shape == model_tensor.shape:
            continue

        if hf_tensor.T.shape == model_tensor.shape:
            print(
                "Needs transpose:",
                hf_key,
                hf_tensor.shape,
                "->",
                model_key,
                model_tensor.shape,
            )
            continue

        print("Missmatched tensor dimension")
        print(hf_key, hf_tensor.shape, "->", model_key, model_tensor.shape)


def copy_weights(hf_state, model_state, weight_map):
    transpose_keys = [
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for hf_key, model_key in weight_map.items():
        hf_tensor = hf_state[hf_key]
        model_tensor = model_state[model_key]

        needs_transpose = any(k in hf_key for k in transpose_keys)

        if needs_transpose:
            model_tensor.copy_(hf_tensor.T)
        else:
            model_tensor.copy_(hf_tensor)


def copy_qkv(hf_state, model_state, n_layers=12, emb_dim=768):
    for i in range(n_layers):
        hf_prefix = f"transformer.h.{i}.attn.c_attn"
        model_prefix = f"transformer_blocks.{i}.att"

        # HuggingFace: [768, 2304] = [Q | K | V]
        qkv_w = hf_state[f"{hf_prefix}.weight"]
        qkv_b = hf_state[f"{hf_prefix}.bias"]
        # Split in correct dimensions
        q_w, k_w, v_w = qkv_w.split(emb_dim, dim=1)
        q_b, k_b, v_b = qkv_b.split(emb_dim, dim=0)

        model_state[f"{model_prefix}.Q.weight"].copy_(q_w.T)
        model_state[f"{model_prefix}.K.weight"].copy_(k_w.T)
        model_state[f"{model_prefix}.V.weight"].copy_(v_w.T)

        model_state[f"{model_prefix}.Q.bias"].copy_(q_b)
        model_state[f"{model_prefix}.K.bias"].copy_(k_b)
        model_state[f"{model_prefix}.V.bias"].copy_(v_b)


if __name__ == "__main__":
    import torch
    import torch_directml
    from Train.train import generate_sample
    import tiktoken

    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    hf_model.eval()

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    hf_state = hf_model.state_dict()
    model_state = model.state_dict()

    copy_weights(hf_state, model_state, weight_map)
    copy_qkv(hf_state, model_state, n_layers=12, emb_dim=768)
    model.load_state_dict(model_state)

    tokenizer = tiktoken.get_encoding("gpt2")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Compare logits output between models
    idx = torch.tensor(
        [tokenizer.encode("Hello i have been loaded from ")], dtype=torch.long
    )

    with torch.no_grad():
        logits_my = model(idx)
        logits_hf = hf_model(idx).logits

    print("Logits difference: ", torch.max(torch.abs(logits_my - logits_hf)))
    # Generate sample text from prelaod weights in custom implementation
    generate_sample(model, tokenizer, device, "Pretained model load completed")
