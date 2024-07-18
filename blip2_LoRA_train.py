import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig, Blip2Config
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

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, processor, answer=True):
        self.processor = processor
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        
        question = self.df["question"][idx]
        answers = [answer["answer"] for answer in self.df["answers"][idx]]
        mode_answer = mode(answers)

        encoding = self.processor(image, question, 
                                  max_length=20, 
                                  padding='max_length', 
                                  truncation=True,
                                  return_tensors="pt")
        
        if 'attention_mask' not in encoding:
            encoding['attention_mask'] = (encoding['input_ids'] != self.processor.tokenizer.pad_token_id).long()

        if self.answer:
            labels = self.processor.tokenizer(mode_answer, 
                                              max_length=20, 
                                              padding='max_length', 
                                              truncation=True)['input_ids']
            encoding["labels"] = torch.tensor(labels)

        for k, v in encoding.items():
            if isinstance(v, list) and isinstance(v[0], np.ndarray):
                encoding[k] = torch.tensor(np.array(v)).squeeze()
            elif isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze()
            else:
                encoding[k] = torch.tensor(v).squeeze()
        return encoding
    
    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    # Collate function to handle batching
    batch_encoding = {k: [item[k] for item in batch if k in item] for k in batch[0].keys()}
    for k, v in batch_encoding.items():
        if k != "answers" and isinstance(v[0], torch.Tensor):
            batch_encoding[k] = torch.stack(v)
    return batch_encoding

def move_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# DataLoader setup
def get_dataloader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=lambda worker_id: torch.manual_seed(worker_id)
    )

# Load BLIP-2 model and processor
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    device_map="auto",
)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


full_train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", processor=processor)

# Calculate sizes for train and validation sets
total_size = len(full_train_dataset)
valid_size = int(0.05 * total_size)  # 10% for validation
train_size = total_size - valid_size

# Create train and validation splits
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

# Set batch size
batch_size = 64

# Create DataLoaders
train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, device=device)
valid_dataloader = get_dataloader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12, device=device)

num_epochs = 15
patience = 5
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

optimizer = AdamW8bit(model.parameters(), lr=5e-4)  # Using 8-bit AdamW optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    
    for idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}')):
        batch = move_to_device(batch, device)
        input_ids = batch['input_ids']
        pixel_values = batch['pixel_values']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        epoch_loss += loss.item() 
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    avg_train_loss = epoch_loss / len(train_dataloader)
    
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(valid_dataloader, desc='Validating'):
            batch = move_to_device(batch, device)
            input_ids = batch['input_ids']
            pixel_values = batch['pixel_values']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                attention_mask=attention_mask,
                                labels=labels)
                eval_loss = outputs.loss
                predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
                predictions = processor.batch_decode(predicted_token_ids, skip_special_tokens=True)
                actual = processor.batch_decode(labels, skip_special_tokens=True)
            correct += sum(p.strip() == a.strip() for p, a in zip(predictions, actual))
            total += len(predictions)

        avg_eval_loss = eval_loss / len(valid_dataloader)
        valid_accuracy = correct / total
        
        print(f"Epoch: {epoch+1} - Train Loss: {avg_train_loss:.4f} - "
              f"Eval Loss: {avg_eval_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f} - "
              f"Valid Accuracy: {valid_accuracy:.4f}")
        
        if avg_eval_loss < min_eval_loss:
            model.save_pretrained("Model/blip2-saved-model", from_pt=True)
            print("Saved model to Model/blip2-saved-model")
            min_eval_loss = avg_eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                print("Early stopping triggered")
                break
    
    scheduler.step()

model.save_pretrained("Model/blip2-saved-model-last", from_pt=True)
print("The finetuning process has finished!")