import torch
import torch.nn.functional as F
from torch.export import export, save

class MinimalModel(torch.nn.Module):
    def __init__(self):
        super(MinimalModel, self).__init__()

    def forward(self):
        queries = torch.randn((1, 32, 48, 128), dtype=torch.float32)
        keys_e = torch.randn((1, 32, 48, 128), dtype=torch.float32)
        values_e = torch.randn((1, 32, 48, 128), dtype=torch.float32)
        attn_mask = None
        dropout_p = 0.0
        is_causal = True

        print(f"SDPA call params - queries: {queries.shape}, keys_e: {keys_e.shape}, values_e: {values_e.shape}, attn_mask: {attn_mask}, dropout_p: {dropout_p}, is_causal: {is_causal}")

        output = F.scaled_dot_product_attention(
            queries,
            keys_e,
            values_e,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Return a dummy second value to meet expected return structure
        dummy_value = torch.tensor(0, dtype=torch.float16)
        return output, dummy_value

def create_reproducible_example():
    model = MinimalModel()
    model.eval()

    try:
        # Run the forward pass first
        print("Running forward pass...")
        output, _ = model()
        print("Forward pass output:", output)

        # Compile and export the model
        print("Exporting the model...")
        compiled_model = torch.compile(model)
        exported_program = export(compiled_model, args=())
        save(exported_program, "exported_model.pt2")
        print("Model export successful")
    except Exception as e:
        print("Error during export:", e)

if __name__ == "__main__":
    create_reproducible_example()
