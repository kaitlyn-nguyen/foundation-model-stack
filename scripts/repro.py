import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, save

class SimpleAttention(nn.Module):
    def __init__(self, emb_dim, nheads, dropout=0.1):
        super(SimpleAttention, self).__init__()
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_length, self.nheads, 3 * self.emb_dim // self.nheads)
        q, k, v = torch.split(qkv, self.emb_dim // self.nheads, dim=-1)

        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, self.nheads, seq_length, seq_length).to(torch.bool)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output.contiguous().view(batch_size, seq_length, self.emb_dim)
        attn_output = self.norm(attn_output)
        return self.out_proj(attn_output)

class MinimalModel(nn.Module):
    def __init__(self, emb_dim=128, nheads=4, max_seq_len=512):
        super(MinimalModel, self).__init__()
        self.embedding = nn.Embedding(1000, emb_dim)
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)
        self.attention = SimpleAttention(emb_dim, nheads)

    def forward(self, x, mask=None):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.position_embedding(position_ids)
        return self.attention(x, mask)

def create_reproducible_example():
    model = MinimalModel()
    model.eval()
    example_input = torch.randint(0, 1000, (1, 10), dtype=torch.long)

    compiled_model = torch.compile(model)

    try:
        exported_program = export(compiled_model, args=(example_input,))
        save(exported_program, "exported_model.pt2")
    except Exception as e:
        print("Error during export:", e)

if __name__ == "__main__":
    create_reproducible_example()
