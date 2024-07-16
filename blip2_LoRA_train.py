import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit
from statistics import mode
import pandas
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np

# Set up device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

# Load BLIP-2 model and processor
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, processor, answer=True):
        self.processor = processor
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        question = self.df["question"][idx]
        answers = [answer["answer"] for answer in self.df["answers"][idx]]
        mode_answer = mode(answers)

        encoding = self.processor(image, question, 
                                  max_length=32, 
                                  padding='max_length', 
                                  truncation=True)
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

# Prepare datasets and dataloaders
full_train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", processor=processor)
total_size = len(full_train_dataset)
valid_size = int(0.01 * total_size)
train_size = total_size - valid_size
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])
test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", processor=processor, answer=False)

batch_size = 8  # Reduced batch size for 8-bit training
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

# Training setup
num_epochs = 15
patience = 10
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

optimizer = AdamW8bit(model.parameters(), lr=2e-5)  # Using 8-bit AdamW optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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
                                     max_new_tokens=16)
            
            predictions = processor.batch_decode(outputs, skip_special_tokens=True)
            actual = processor.batch_decode(labels, skip_special_tokens=True)
            
            correct += sum(p.strip() == a.strip() for p, a in zip(predictions, actual))
            total += len(predictions)
    
    return correct / total

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for idx, batch in tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch+1} Training'):
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=labels)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    model.eval()
    eval_loss = 0
    for idx, batch in tqdm(enumerate(valid_dataloader), desc='Validating'):
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
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
        model.save_pretrained("Model/blip2-lora-model")
        print("Saved model to Model/blip2-lora-model")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break

pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has completed!")

# Generate submission file
model.eval()
submission = []
for idx, batch in tqdm(enumerate(test_dataloader), desc="Generating predictions"):
    input_ids = batch["input_ids"].to(device)
    pixel_values = batch["pixel_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    out = model.generate(input_ids=input_ids,
                         pixel_values=pixel_values,
                         attention_mask=attention_mask,
                         max_new_tokens=16)
    decoded_outputs = processor.batch_decode(out, skip_special_tokens=True)
    cleaned_outputs = [pred.strip().lower() for pred in decoded_outputs]
        
    submission.extend(cleaned_outputs)

predictions_array = np.array(submission)
np.save("submission.npy", predictions_array)

print("Submission file created.")