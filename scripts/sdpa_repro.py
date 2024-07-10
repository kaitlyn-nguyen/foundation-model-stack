import torch
import torch.nn as nn
from torch.export import export, save

class DummyQKV(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DummyQKV, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_length, self.num_heads, 3 * embed_dim // self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

class DummyModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DummyModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = DummyQKV(embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        q, k, v = self.qkv(x)

        # Transpose for scaled dot product attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True)

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        return output

embed_dim = 64
num_heads = 8
dummy_model = DummyModel(embed_dim, num_heads)

input_tensor = torch.randn(32, 10, embed_dim)  # Batch size: 32, Sequence length: 10, Embedding size: 64

# Try export and save the model
try:
    exported_program = export(dummy_model, (input_tensor,))
    save(exported_program, "dummy_model_export.pt2")
    print("Saved model successfully")
except Exception as e:
    print("Error during export:", e)
    raise
