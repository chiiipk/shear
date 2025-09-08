# utils/shape_aware_loader.py
import os
import torch
import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel, AutoTokenizer

def _load_state_dict_from_dir(model_dir: str):
    pt = os.path.join(model_dir, "pytorch_model.bin")
    st = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st, device="cpu")
    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu")
    raise FileNotFoundError(f"Không tìm thấy weight file trong {model_dir}")

def _replace_linear(lin: nn.Linear, out_features=None, in_features=None) -> nn.Linear:
    out_features = lin.out_features if out_features is None else int(out_features)
    in_features  = lin.in_features  if in_features  is None else int(in_features)
    return nn.Linear(in_features, out_features, bias=(lin.bias is not None))

@torch.no_grad()
def load_xlmr_clean_pruned(model_dir: str, device: str | None = None, dtype: torch.dtype = torch.float32):
    """
    Loader cho checkpoint 'clean-pruned':
      - Đọc shape q/k/v & attention.output.dense từ state_dict
      - Co module Linear tương ứng trong model khởi tạo
      - Cập nhật attention_head_size / all_head_size
      - Nạp state_dict đầy đủ (không init random)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = XLMRobertaConfig.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = XLMRobertaModel(cfg)

    sd = _load_state_dict_from_dir(model_dir)

    L = cfg.num_hidden_layers
    for i in range(L):
        base = f"encoder.layer.{i}.attention"
        # Các tensor phải tồn tại trong checkpoint đã export
        qw = sd[f"{base}.self.query.weight"]       # [q_out, hidden]
        kw = sd[f"{base}.self.key.weight"]
        vw = sd[f"{base}.self.value.weight"]
        ow = sd[f"{base}.output.dense.weight"]     # [hidden, q_out]

        q_out, hidden = int(qw.shape[0]), int(qw.shape[1])
        assert kw.shape[0] == q_out and vw.shape[0] == q_out, f"Layer {i}: Q/K/V out_features không đồng nhất"
        assert ow.shape[0] == hidden and ow.shape[1] == q_out, f"Layer {i}: out-proj shape không khớp"

        attn = model.encoder.layer[i].attention
        # 1) Co Q/K/V: out=q_out, in=hidden
        attn.self.query = _replace_linear(attn.self.query, out_features=q_out, in_features=hidden)
        attn.self.key   = _replace_linear(attn.self.key,   out_features=q_out, in_features=hidden)
        attn.self.value = _replace_linear(attn.self.value, out_features=q_out, in_features=hidden)
        # 2) Co out-proj: in=q_out, out=hidden
        attn.output.dense = _replace_linear(attn.output.dense, out_features=hidden, in_features=q_out)

        # 3) Cập nhật metadata head dims theo num_heads trong config
        num_heads = attn.self.num_attention_heads
        if q_out % num_heads != 0:
            raise ValueError(f"Layer {i}: q_out={q_out} không chia hết cho num_heads={num_heads}.")
        head_dim = q_out // num_heads
        attn.self.attention_head_size = head_dim
        if hasattr(attn.self, "all_head_size"):
            attn.self.all_head_size = q_out

    # Nạp state_dict sau khi “co” module
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print("⚠️ unexpected keys:", unexpected)
    if missing:
        print("⚠️ missing keys:", missing)

    return model.to(device=device, dtype=dtype), tok
