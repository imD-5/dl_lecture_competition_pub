import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from peft.utils.other import prepare_model_for_kbit_training
from peft.mapping import get_peft_model
from peft.mapping import LoraConfig
from bitsandbytes.optim import AdamW8bit
from statistics import mode
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np

# Set up device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16
)

# Load BLIP-2 model and processor
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    quantization_config=bnb_config,
    device_map="auto"
)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "query",
        "key",
        "value",
        "output.dense",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(model)