from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from torch.export import export

# Load the model
model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

# Prepare the input text
input_text = "Hello, I am"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
tokens = model.generate(
    **inputs,
    max_length=50,  # Maximum length of the generated sequence
    num_return_sequences=1,  # Number of sequences to generate
    do_sample=True,  # Whether or not to use sampling
    temperature=0.7,  # Sampling temperature
    top_k=50,  # Keep only top k tokens with highest probability
    top_p=0.95,  # Keep only tokens with cumulative probability >= top_p
)

# Decode and print the generated text
generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
print("First generation:", generated_text)

# Define a simple wrapper module for exporting
class MyWrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(MyWrapperModel, self).__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

# Example inputs for export
example_inputs = (
    torch.randint(0, 50256, (1, 10)),  # Example input_ids
    torch.ones(1, 10)  # Example attention_mask
)

# Export the model
exported_program = export(
    MyWrapperModel(model),
    args=example_inputs
)

# Save the exported model
torch.export.save(exported_program, 'exported_program.pt2')

print("Model exported successfully")

# Load the exported model
loaded_exported_program = torch.export.load('exported_program.pt2')

# Prepare new input for inference
new_input_text = "Once upon a time"
new_inputs = tokenizer(new_input_text, return_tensors="pt")

# Pad new inputs to match the export sequence length (10)
padded_input_ids = torch.cat([new_inputs['input_ids'], torch.zeros(1, 10 - new_inputs['input_ids'].shape[1], dtype=torch.long)], dim=1)
padded_attention_mask = torch.cat([new_inputs['attention_mask'], torch.zeros(1, 10 - new_inputs['attention_mask'].shape[1], dtype=torch.long)], dim=1)

# Run inference with the loaded exported model
loaded_model = loaded_exported_program.module()  # Get the underlying model
loaded_model_output = loaded_model(
    padded_input_ids, padded_attention_mask
)

# Decode and print the generated text from the loaded model
loaded_generated_text = tokenizer.decode(loaded_model_output[0][0], skip_special_tokens=True)
print("Generated text from loaded model:", loaded_generated_text)
