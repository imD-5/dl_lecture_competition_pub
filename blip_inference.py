import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.df['image'][idx]))
        question = self.df["question"][idx]
        
        encoding = self.transform(image, question, 
                                  max_length=16, 
                                  padding='max_length', 
                                  truncation=True,
                                  return_tensors="pt")
        
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        
        return encoding

    def __len__(self):
        return len(self.df)

def load_model(model_path):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(model_path)
    return processor, model

def create_submission(model, processor, test_dataloader, device):
    model.eval()
    submission = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            out = model.generate(input_ids=input_ids,
                                 pixel_values=pixel_values,
                                 attention_mask=attention_mask,
                                 max_new_tokens=20)  # Adjust max_new_tokens as needed
            
            decoded_outputs = processor.batch_decode(out, skip_special_tokens=True)
            cleaned_outputs = [pred.strip().lower() for pred in decoded_outputs]
            submission.extend(cleaned_outputs)
    
    return submission

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model_path = "Model/blip-saved-model"  # Adjust this path to your saved model
    processor, model = load_model(model_path)
    model.to(device)

    # Create test dataset and dataloader
    test_dataset = VQADataset(df_path="./data/valid.json", 
                              image_dir="./data/valid", 
                              transform=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Generate submission
    submission = create_submission(model, processor, test_dataloader, device)

    # Save submission
    predictions_array = np.array(submission)
    np.save("submission.npy", predictions_array)
    print(f"Submission saved to submission.npy with {len(predictions_array)} predictions.")

if __name__ == "__main__":
    main()