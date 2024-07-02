import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.export import export, save, load

parser = argparse.ArgumentParser()
parser.add_argument('--strict', action=argparse.BooleanOptionalAction)
parser.add_argument('--pre', action=argparse.BooleanOptionalAction)
args = parser.parse_args()
print(f"Strict mode: {args.strict}")
print(f"Pre-dispatch mode: {args.pre}")

# Grab the model
with torch.device("meta"):
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf"
    )
    llama.eval()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token

    prompts = (
        "How do you", "I like to",
    )

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

if args.pre:
    ep = export(
        llama,
        (inputs["input_ids"],),
        strict=args.strict,
    )
else:
    _export(
        llama,
        (inputs["input_ids"],),
        pre_dispatch=args.pre,
        strict=args.strict,
    )