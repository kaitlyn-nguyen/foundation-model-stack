import argparse
import itertools
import os
import random
import sys
import logging
import sys

import numpy as np
import torch
import torch._inductor.config
from torch import distributed as dist
from torch.export import export, save, load

from fms.models import get_model
from fms.utils import generation, tokenizers
from fms.utils.generation import generate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument("--context_file", type=str, default=None, help="File to summarize")

parser.add_argument(
    "--export_model",
    action="store_true",
    help="Export the compiled model using torch.export",
)
parser.add_argument(
    "--export_path",
    type=str,
    default="exported_model.pt2",
    help="Path to save the exported model",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.float16)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

logger.info("Loading model...")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
logger.info(f"Model loading complete on rank {local_rank}")

past_key_value_states = [(torch.zeros((1, 32, 48, 128), dtype=torch.float16, device=device), 
                          torch.zeros((1, 32, 48, 128), dtype=torch.float16, device=device)) for _ in range(32)]

class ForwardModule(torch.nn.Module):
    def __init__(self, model):
        super(ForwardModule, self).__init__()
        self.model = model

    def forward(self, input_ids, past_key_value_states=None):
        return self.model.forward(input_ids, attn_algorithm="math", past_key_value_states=past_key_value_states, use_cache=True) 

def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    tokens = ["<s>"] + tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids

def pad_prompt(prompt, pad_len, pad_token="<unk>"):
    to_pad = pad_len - len(prompt)
    if to_pad == 0:
        return prompt

    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    pad_ids = [pad_id] * to_pad
    return torch.cat((torch.tensor(pad_ids, device=device), prompt))

if args.context_file is not None:
    with open(args.context_file) as file:
        long_prompt = file.read()
        prompt1 = (
            long_prompt
            + "\nPlease give me a brief summary of this research paper in a few bullet points."
        )
        prompt2 = long_prompt + "\nPlease write me the abstract for this paper."
else:
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

    prompt1 = template.format(
        "Provide a list of instructions for preparing egg tart."
    )
    prompt2 = template.format("Explain some popular greetings in Spanish.")

prompt1 = ids_for_prompt(prompt1)
prompt2 = ids_for_prompt(prompt2)

max_len = max([len(prompt) for prompt in [prompt1, prompt2]])

ids = prompt1.unsqueeze(0)

if args.compile:
    logger.info("Compiling model...")
    model = torch.compile(model, mode=args.compile_mode)


# Create forward module instance
forward_module = ForwardModule(model)

# Measure normal forward call time
normal_start_event = torch.cuda.Event(enable_timing=True)
normal_end_event = torch.cuda.Event(enable_timing=True)

normal_start_event.record()
with torch.no_grad():
    normal_outputs = model.forward(ids, attn_algorithm="math", past_key_value_states=past_key_value_states, use_cache=True)
normal_end_event.record()

torch.cuda.synchronize()
normal_forward_time = normal_start_event.elapsed_time(normal_end_event)
logger.info(f"Eager mode forward call time: {normal_forward_time} ms")

# Measure compilation time of a forward call
compile_start_event = torch.cuda.Event(enable_timing=True)
compile_end_event = torch.cuda.Event(enable_timing=True)

compile_start_event.record()
compiled_model_fake = torch.compile(forward_module.forward)
compile_end_event.record()


torch.cuda.synchronize()
compile_time = compile_start_event.elapsed_time(compile_end_event)
logger.info(f"Compilation time of a forward call: {compile_time} ms")
compiled_model = torch.compile(forward_module)

# Measure compiled forward call time
compiled_start_event = torch.cuda.Event(enable_timing=True)
compiled_end_event = torch.cuda.Event(enable_timing=True)

compiled_start_event.record()
with torch.no_grad():
    compiled_outputs = compiled_model.forward(ids, past_key_value_states=past_key_value_states)
compiled_end_event.record()

torch.cuda.synchronize()
compiled_forward_time = compiled_start_event.elapsed_time(compiled_end_event)
logger.info(f"Compiled forward call time: {compiled_forward_time} ms")

# Export the forward call
export_start_event = torch.cuda.Event(enable_timing=True)
export_end_event = torch.cuda.Event(enable_timing=True)

export_start_event.record()

#export starts here
exported_program = export(forward_module, args=(ids, past_key_value_states))
save(exported_program, args.export_path)
export_end_event.record()

torch.cuda.synchronize()
export_time = export_start_event.elapsed_time(export_end_event)
logger.info(f"Export time of a forward call: {export_time} ms")

# Load the exported forward call
load_start_event = torch.cuda.Event(enable_timing=True)
load_end_event = torch.cuda.Event(enable_timing=True)

load_start_event.record()
loaded_program = load(args.export_path).module()
load_end_event.record()

torch.cuda.synchronize()
load_time = load_start_event.elapsed_time(load_end_event)
logger.info(f"Load time: {load_time} ms")

# Measure forward call time exported and loaded mode
loaded_forward_start_event = torch.cuda.Event(enable_timing=True)
loaded_forward_end_event = torch.cuda.Event(enable_timing=True)


loaded_forward_start_event.record()
with torch.no_grad():
    loaded_forward_outputs = loaded_program.forward(ids, past_key_value_states)
loaded_forward_end_event.record()

torch.cuda.synchronize()
loaded_forward_time = loaded_forward_start_event.elapsed_time(loaded_forward_end_event)
logger.info(f"Exported and loaded mode forward call: {loaded_forward_time} ms")

def print_result(result):
    if local_rank != 0:
        return
    result = generation.truncate_after_eos(result, tokenizer.eos_token_id)
    logger.info(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))

def infer(use_cache, do_sample):
    if local_rank == 0:
        logger.info(f"use_cache {use_cache} ;; do_sample {do_sample}")
        logger.info("==================")

    result = generate(
        model,
        ids,
        max_new_tokens=200,
        use_cache=use_cache,
        do_sample=do_sample,
        max_seq_len=2000,
    )
    for i in range(result.shape[0]):
        print_result(result[i])

logger.info(f"Generating output on rank {local_rank}")
do_sample = [False]
use_cache = [args.no_use_cache]
for sample, cache in itertools.product(do_sample, use_cache):
    infer(cache, sample)
