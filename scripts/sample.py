import torch
from torch import nn
from torch.export import export, save, load
import time

# Define a more complex LLM model
class ComplexLLM(nn.Module):
    def __init__(self):
        super(ComplexLLM, self).__init__()
        self.embedding = nn.Embedding(10000, 128)
        self.rnn = nn.LSTM(128, 256, batch_first=True, num_layers=2)
        self.fc = nn.Linear(256, 10000)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Create example input
example_input = torch.randint(0, 10000, (32, 50))  # Batch size 32, sequence length 50

# Export the model
model = ComplexLLM()
exported_program = export(model, (example_input,))

# Save the exported model
save(exported_program, 'exported_complex_llm.pt2')

# Load the exported model
loaded_exported_program = load('exported_complex_llm.pt2')

# Get the module from the exported program
exported_model = loaded_exported_program.module()

# Compile the model
compiled_model = torch.compile(model)

# Export the compiled model
compiled_exported_program = export(compiled_model, (example_input,))

# Save the compiled exported model
save(compiled_exported_program, 'compiled_exported_complex_llm.pt2')

# Load the compiled exported model
loaded_compiled_exported_program = load('compiled_exported_complex_llm.pt2')

# Get the module from the compiled exported program
compiled_exported_model = loaded_compiled_exported_program.module()

# Define a function to measure inference time
def measure_inference_time(model, input, iterations=100):
    # Warm-up
    for _ in range(10):
        model(input)

    # Measure time
    start_time = time.time()
    for _ in range(iterations):
        model(input)
    end_time = time.time()
    
    avg_time_per_inference = (end_time - start_time) / iterations
    return avg_time_per_inference

# Define a function to generate tokens
def generate_tokens(model, input_seq, num_tokens_to_generate):
    model.eval()
    generated_tokens = input_seq.tolist()[0]
    
    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            input_tensor = torch.tensor(generated_tokens[-50:]).unsqueeze(0)  # Use the last 50 tokens
            output = model(input_tensor)
            next_token = torch.argmax(output, dim=-1).item()
            generated_tokens.append(next_token)
    
    return generated_tokens

# Define a function to measure the time taken to generate a sequence of tokens
def measure_generation_time(model, input_seq, num_tokens_to_generate):
    model.eval()
    generated_tokens = input_seq.tolist()[0]
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            input_tensor = torch.tensor(generated_tokens[-50:]).unsqueeze(0)  # Use the last 50 tokens
            output = model(input_tensor)
            next_token = torch.argmax(output, dim=-1).item()
            generated_tokens.append(next_token)
    end_time = time.time()
    
    total_generation_time = end_time - start_time
    return total_generation_time

# Measure inference time for the original model
original_inference_time = measure_inference_time(model, example_input)
print(f'Average inference time for the original model: {original_inference_time:.6f} seconds')

# Measure inference time for the compiled model
compiled_inference_time = measure_inference_time(compiled_model, example_input)
print(f'Average inference time for the compiled model: {compiled_inference_time:.6f} seconds')

# Measure inference time for the exported model
exported_inference_time = measure_inference_time(exported_model, example_input)
print(f'Average inference time for the exported model: {exported_inference_time:.6f} seconds')

# Measure inference time for the compiled exported model
compiled_exported_inference_time = measure_inference_time(compiled_exported_model, example_input)
print(f'Average inference time for the compiled exported model: {compiled_exported_inference_time:.6f} seconds')