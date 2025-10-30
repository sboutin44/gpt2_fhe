"""
Step-by-Step Guide to Running GPT-2 with Fully Homomorphic Encryption (FHE)

This example demonstrates how to run a quantized version of GPT-2's attention
mechanism on encrypted data using Concrete ML.
"""

# ============================================================================
# STEP 1: Setup and Imports
# ============================================================================

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# These would be your custom modules from the provided code
from qgpt2_models import SingleHeadQGPT2Model, MultiHeadsQGPT2Model
# from load_huggingface import get_gpt2_model, get_gpt2_tokenizer

print("Step 1: Imports complete")


# ============================================================================
# STEP 2: Load Pre-trained GPT-2 Model (Standard Version)
# ============================================================================

# Load the standard GPT-2 model for comparison
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Step 2: Loaded standard GPT-2 model")


# ============================================================================
# STEP 3: Test with Standard GPT-2 (Baseline)
# ============================================================================

input_sentence = "Computations on encrypted data can help"
input_ids = gpt2_tokenizer.encode(input_sentence, return_tensors="pt")

# Generate with standard model
with torch.no_grad():
    output_ids = gpt2_model.generate(input_ids, max_new_tokens=4)
    baseline_output = gpt2_tokenizer.decode(output_ids[0])

print(f"Step 3: Baseline output: '{baseline_output}'")

# ============================================================================
# STEP 4: Load Quantized FHE-Compatible Model
# ============================================================================

# Option A: Single Head Attention (faster, less accurate)
# This replaces ONLY the first attention head of the first layer with FHE ops
single_head_model = SingleHeadQGPT2Model.from_pretrained(
    "gpt2",
    n_bits=7,  # Use 7-bit quantization (balances accuracy and performance)
    layer=0,   # Apply to first layer
    use_cache=False
)

# Option B: Multi-Head Attention (slower, more accurate)
# This replaces the ENTIRE multi-head attention of the first layer with FHE ops
# multi_head_model = MultiHeadsQGPT2Model.from_pretrained(
#     "gpt2",
#     n_bits=7,  # Use 7-bit quantization
#     layer=0,   # Apply to first layer
#     use_cache=False
# )


# ============================================================================
# STEP 5: Run in Clear Mode (Quantized but Not Encrypted)
# ============================================================================

# This tests the quantized model without encryption
model = single_head_model  # or multi_head_model
model.set_fhe_mode(fhe="disable")  # Run quantized operations in the clear

output_ids_clear = model.generate(input_ids, max_new_tokens=4)
clear_output = gpt2_tokenizer.decode(output_ids_clear[0])

print(f"Step 5: Quantized (clear) output: '{clear_output}'")

# ============================================================================
# STEP 6: Compile the Model for FHE Execution
# ============================================================================

# This step:
# 1. Runs forward pass to calibrate quantization parameters
# 2. Builds an FHE circuit that can operate on encrypted data
# 3. Analyzes bit-width requirements

circuit = model.compile(
    input_ids,
    msbs_round=6  # Rounding parameter to reduce bit-width
)

print(f"Step 6: Circuit compiled!")
print(f"  Max bit-width: {circuit.graph.maximum_integer_bit_width()} bits")



# ============================================================================
# STEP 8: Execute in Full FHE Mode (On Encrypted Data!)
# ============================================================================

# This is the actual FHE execution where computations happen on encrypted data
# WARNING: This is SLOW (can take minutes per forward pass)

import time

model.set_fhe_mode(fhe="execute")

start = time.time()
output_logits_fhe = model(input_ids).logits
execution_time = time.time() - start

predicted_token_fhe = torch.argmax(output_logits_fhe[0, -1, :]).item()
predicted_word_fhe = gpt2_tokenizer.decode([predicted_token_fhe])

print(f"Step 8: FHE execution complete!")
print(f"  Time taken: {execution_time:.2f} seconds")
print(f"  Predicted word: '{predicted_word_fhe}'")
