import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define your Hugging Face token
huggingface_token = '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'

# Path to the LLaMA model
llama_path = "meta-llama/Llama-2-7b-hf"

# Load the model and tokenizer with authentication
model = AutoModelForCausalLM.from_pretrained(llama_path, use_auth_token=huggingface_token).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token=huggingface_token)

# Prepare dummy input
example_inputs = tokenizer("This is a test input", return_tensors="pt").to('cuda')

# Export the model
print("Exporting the LLaMA model...")
exported_program = torch.export(model, args=(example_inputs.input_ids,))
torch.save(exported_program, "llama_7b_model.pt2")

# Load the exported model
loaded_model = torch.load("llama_7b_model.pt2").module()

# Test the loaded model
print(loaded_model.generate(input_ids=example_inputs.input_ids))
