# import torch
# from torch.export import export 

# class ToyModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(10, 10)
#         self.fc2 = torch.nn.Linear(10, 10)
#         self.fc3 = torch.nn.Linear(10, 10)
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
#         a = torch.sin(x)
#         b = torch.cos(y)
#         y = self.fc1(x)
#         return a + b + y
    
# f = ToyModule()

# example_args = (torch.randn(10, 10), torch.randn(10, 10))
# out_ref = f(*example_args)

# exported_program: torch.export.ExportedProgram = export(
#     f, args=example_args
# )
# print("Here is program", exported_program) 


# out = exported_program.module()(*example_args)
# assert torch.allclose(out, out_ref)


from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set your Hugging Face API token here
# Make sure it has READ access to avoid connection issues
api_token="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"

# Set the cache directory outside your home directory!
mydir = "/mnt/c/Users/KaitlynNguyen/Documents/KN-FMS"
os.environ["TRANSFORMERS_CACHE"] = mydir
os.environ["HUGGINGFACE_HUB_CACHE"] = mydir

# pre-trained model we will use 
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

try:
    # Load the tokenizer and model with authentication
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token, cache_dir=mydir)

    # with the device_map="auto" arg, the transform library distribute the model layers accross the different GPUs detected
    model = AutoModelForCausalLM.from_pretrained(model_name, token=api_token, device_map="auto", cache_dir=mydir)

    # Example inference
    input_text = "Can you recommend me a restaurant around here? I am new in town."
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

except Exception as e:
    print(f"An error occurred: {e}")