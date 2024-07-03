import torch
from torch import nn
from torch.export import export

class SimpleLLM(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(10000, embed_dim)  
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, *, mask=None) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(0, 1)  
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x.transpose(0, 1)  

# Function to compile and export the model with different batch sizes, prompt sizes, and max new tokens
def compile_and_export_model():
    batch_sizes = [1, 2, 4, 8]
    prompt_sizes = [128, 256, 512]
    max_new_tokens_list = [50, 100, 200]

    model = SimpleLLM().cuda()
    
    for batch_size in batch_sizes:
        for prompt_size in prompt_sizes:
            for max_new_tokens in max_new_tokens_list:
                example_args = (torch.randint(0, 10000, (batch_size, prompt_size)).cuda(),)
                example_kwargs = {"mask": None}
                
                # Compile the model
                compiled_model = torch.compile(model)
                
                # Forward pass to trigger compilation
                with torch.no_grad():
                    compiled_model(*example_args, **example_kwargs)
                
                # Export the compiled model
                exported_program = export(compiled_model, args=example_args, kwargs=example_kwargs)
                
                # Save the exported model
                export_path = f"compiled_model_bs{batch_size}_ps{prompt_size}_mnt{max_new_tokens}.pt2"
                torch.save(exported_program, export_path)
                print(f"Exported compiled model saved as {export_path}")

compile_and_export_model()
