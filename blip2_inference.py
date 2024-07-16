import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, processor):
        self.processor = processor
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.df['image'][idx]))
        question = self.df["question"][idx]
        
        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        
        return encoding

    def __len__(self):
        return len(self.df)

def load_model(model_path):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    return processor, model

def generate_submissions(model, processor, test_loader, device):
    model.eval()
    submissions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=20
            )
            
            pred = processor.decode(outputs[0], skip_special_tokens=True)
            submissions.append(pred)
    
    return submissions

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model_path = "path/to/your/trained/model"  # Update this path
    processor, model = load_model(model_path)
    model.to(device)

    # Create test dataset and dataloader
    test_dataset = VQADataset(
        df_path="./data/valid.json",
        image_dir="./data/valid",
        processor=processor
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Generate submissions
    submissions = generate_submissions(model, processor, test_loader, device)

    # Save submissions
    submissions_array = np.array(submissions)
    np.save("submission.npy", submissions_array)
    print(f"Submission file created: submission.npy")
    print(f"Total predictions: {len(submissions_array)}")

if __name__ == "__main__":
    main()