from transformers import BlipProcessor, BlipForQuestionAnswering
from torch.utils.data import DataLoader, Dataset, random_split
from statistics import mode
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np
from collections import Counter
from typing import List, Tuple

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
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
                                  max_length=16, 
                                  padding='max_length', 
                                  truncation=True,
                                  return_tensors="pt")
        
        if 'attention_mask' not in encoding:
            encoding['attention_mask'] = (encoding['input_ids'] != self.processor.tokenizer.pad_token_id).long()

        if self.answer:
            labels = self.processor.tokenizer(mode_answer, 
                                              max_length=16, 
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

def unfreeze_layers(epoch):
    if epoch == 0:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        for param in model.text_decoder.parameters():
            param.requires_grad = False
    elif epoch == 5:
        for param in model.text_decoder.parameters():
            param.requires_grad = True
    elif epoch == 25:
        for param in model.vision_model.parameters():
            param.requires_grad = True

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating accuracy'):
            batch = move_to_device(batch, device)
            input_ids = batch['input_ids']
            pixel_values = batch['pixel_values']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model.generate(input_ids=input_ids,
                                     pixel_values=pixel_values,
                                     attention_mask=attention_mask,
                                     max_new_tokens=8)
            
            predictions = processor.batch_decode(outputs, skip_special_tokens=True)
            actual = processor.batch_decode(labels, skip_special_tokens=True)
            
            correct += sum(p.strip() == a.strip() for p, a in zip(predictions, actual))
            total += len(predictions)
    
    return correct / total

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: List[Tuple[List[str], List[float]]]):
    total_acc = 0.
    batch_size = batch_pred.shape[0]
    for i in range(batch_size):
        pred = batch_pred[i]
        answers, weights = batch_answers[i]
        acc = 0.
        for j, answer in enumerate(answers):
            if pred == answer:
                acc += weights[j]
        total_acc += min(acc, 1.0)
    return total_acc / batch_size

full_train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", processor=processor)

# Calculate sizes for train and validation sets
total_size = len(full_train_dataset)
valid_size = int(0.01 * total_size)  # 10% for validation
train_size = total_size - valid_size

# Create train and validation splits
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

# Set batch size
batch_size = 64

# Create DataLoaders
train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, device=device)
valid_dataloader = get_dataloader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12, device=device)

num_epochs = 20
patience = 5
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
print("train len ", len(train_dataloader ))

# Unfreeze layers once before training
unfreeze_layers(0)

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
        
        avg_eval_loss = eval_loss / len(valid_dataloader)
        valid_accuracy = calculate_accuracy(model, valid_dataloader, device)
        
        print(f"Epoch: {epoch+1} - Train Loss: {avg_train_loss:.4f} - "
              f"Eval Loss: {avg_eval_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f} - "
              f"Valid Accuracy: {valid_accuracy:.4f}")
        
        if avg_eval_loss < min_eval_loss:
            model.save_pretrained("Model/blip-saved-model", from_pt=True)
            print("Saved model to Model/blip-saved-model")
            min_eval_loss = avg_eval_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                print("Early stopping triggered")
                break
    
    scheduler.step()

    # Update unfrozen layers less frequently
    if epoch in [5]:
        unfreeze_layers(epoch)

print("The finetuning process has finished!")

model.save_pretrained("Model/blip-saved-model-last", from_pt=True)