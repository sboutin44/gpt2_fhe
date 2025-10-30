import torch
from transformers import GPT2Tokenizer, GPT2Config
from qgpt2_models import SingleHeadQGPT2Model, MultiHeadsQGPT2Model

config = GPT2Config.from_pretrained("gpt2")
model = SingleHeadQGPT2Model(config, n_bits=16, layer=0)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello world!"
inputs = tokenizer(text, return_tensors="pt")["input_ids"]

# Run model in "clear" mode for calibration
model.set_fhe_mode(fhe="disable")
outputs = model(inputs)

# Compile the calibrated model to an FHE circuit
model.compile(inputs)

model.set_fhe_mode("execute")
outputs_fhe = model(inputs)
