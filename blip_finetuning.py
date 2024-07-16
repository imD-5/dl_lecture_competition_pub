import os

import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from torch.utils.data import DataLoader, random_split
from statistics import mode
import pandas
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import numpy as np

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)

torch.backends.cudnn.enabled = False

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        
        question = self.df["question"][idx]
        answers = [answer["answer"] for answer in self.df["answers"][idx]]
        mode_answer = mode(answers)

        encoding = self.transform(image, question, 
                                  max_length=16, 
                                  padding='max_length', 
                                  truncation=True)
        if self.answer:
            labels = self.transform.tokenizer(mode_answer, 
                                              max_length=8, 
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

full_train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=processor)

# Calculate sizes for train and validation sets
total_size = len(full_train_dataset)
valid_size = int(0.01 * total_size)  # 10% for validation
train_size = total_size - valid_size

# Create train and validation splits
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

# Load the test dataset
test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=processor, answer=False)

# Set batch size
batch_size = 32

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)


optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

num_epochs = 100
patience = 10
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

print("train len ", len(train_dataloader ))
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating accuracy'):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.generate(input_ids=input_ids,
                                     pixel_values=pixel_values,
                                     attention_mask=attention_mask,
                                     max_new_tokens=8)
            
            predictions = processor.batch_decode(outputs, skip_special_tokens=True)
            actual = processor.batch_decode(labels, skip_special_tokens=True)
            
            correct += sum(p.strip() == a.strip() for p, a in zip(predictions, actual))
            total += len(predictions)
    
    return correct / total

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for idx, batch in tqdm(enumerate(train_dataloader), desc='Training batch'):

        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    model.eval()
    eval_loss = 0
    for idx, batch in tqdm(enumerate(valid_dataloader), desc='Validating batch'):
        if idx >= 20 :
              break
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_masked,
                        labels=labels)
        
        loss = outputs.loss
        eval_loss += loss.item()


    valid_accuracy = calculate_accuracy(model, valid_dataloader, device)

    tracking_information.append((epoch_loss/len(train_dataloader), 
                                 eval_loss/len(valid_dataloader), 
                                 optimizer.param_groups[0]["lr"],
                                 valid_accuracy))
    
    print(f"Epoch: {epoch+1} - Training loss: {epoch_loss/len(train_dataloader):.4f} - "
          f"Eval Loss: {eval_loss/len(valid_dataloader):.4f} - LR: {optimizer.param_groups[0]['lr']:.6f} - "
          f"Valid Accuracy: {valid_accuracy:.4f}")
    
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained("Model/blip-saved-model", from_pt=True) 
        print("Saved model to Model/blip-saved-model")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break

pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")

# 提出用ファイルの作成
model.eval()
submission = []
for idx, batch in tqdm(enumerate(test_dataloader)):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    attention_masked = batch.pop('attention_mask').to(device)
    labels = batch.pop('labels').to(device)
    
    out = model.generate(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_masked,
                        labels=labels)
    decoded_outputs = processor.decode(out[0], skip_special_tokens=True)
    # Optional: Clean up the predictions (e.g., strip whitespace, lowercase)
    cleaned_outputs = [pred.strip().lower() for pred in decoded_outputs]
        
    submission.extend(cleaned_outputs)

# Convert list of predictions to numpy array and save
predictions_array = np.array(submission)
np.save("submission.npy", predictions_array)

submission = np.array(submission)
torch.save(model.state_dict(), "model.pth")
np.save("submission.npy", submission)

