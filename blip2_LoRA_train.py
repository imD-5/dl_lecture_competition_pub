import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import time
from statistics import mode

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, processor, answer=True):
        self.processor = processor
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        question = self.df["question"][idx]
        
        if self.answer:
            answers = [answer["answer"] for answer in self.df["answers"][idx]]
            mode_answer = mode(answers)
            encoding = self.processor(image, question, mode_answer, padding="max_length", truncation=True, return_tensors="pt")
        else:
            encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        
        return encoding

    def __len__(self):
        return len(self.df)

def train(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        labels = batch.pop("labels").to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        load_in_8bit=True,
        device_map="auto"
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", processor=processor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    num_epochs = 20
    optimizer = AdamW8bit(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, device, scaler)
        end_time = time.time()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Time: {end_time - start_time:.2f} seconds")

    # Save the LoRA model
    model.save_pretrained("blip2_lora_model")
    print("Training completed. LoRA model saved.")

if __name__ == "__main__":
    main()