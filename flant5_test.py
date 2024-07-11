import torch.nn as nn
import torch

from transformers import T5ForConditionalGeneration,  AutoTokenizer
from transformers import ViTMAEModel
from torchvision import transforms
from PIL import Image

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        return out
    
class MultimodalVAE(nn.Module):
    def __init__(self, device):
        super().__init__()  
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", device_map="auto")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(self.device)
        
        self.cross_attention = CrossAttention(768).to(self.device)
    
    def get_initial_decoder_input_ids(self, tokenizer, batch_size=1):
        # Get the ID of the pad token, which T5 uses as the start token
        pad_token_id = tokenizer.pad_token_id
        
        # Create a tensor with the pad token ID (start token for T5)
        # Shape: (batch_size, 1)
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            pad_token_id, 
            dtype=torch.long
        )
    
        return decoder_input_ids
        
    def forward(self, text_input, image_input):
        
        text_tokens = self.tokenizer(text_input, return_tensors="pt").input_ids.to(self.device)
        encoded_text = self.t5_model.encoder(text_tokens)[0].to(self.device)
        
        image_inputs = self.image_transforms(image_input).unsqueeze(0).to(self.device)
        encoded_image = self.image_encoder(image_inputs).last_hidden_state.to(self.device)
        
        print(f"Encoded text shape: {encoded_text}")
        print(f"Encoded image shape: {encoded_image}")
        
        #cross_attended_text = self.cross_attention(encoded_text, encoded_image)
        #combined_features = torch.cat([encoded_text, cross_attended_text], dim=1)

        decoder_input_ids = self.get_initial_decoder_input_ids(self.tokenizer).to(self.device)
        for _ in range(50):
            decoder_outputs = self.t5_model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoded_text,
                use_cache=False,
                return_dict=True
            )
            sequence_output = decoder_outputs.last_hidden_state
            lm_logits = self.t5_model.lm_head(sequence_output)
            
            # Get the most likely next token
            next_token = torch.argmax(lm_logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            # Append to the growing sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            
            # Stop if we predict the end of sequence token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode the generated sequence
        decoded_output = self.tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        return decoded_output
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MMEncoder =  MultimodalVAE(device)
input_text = "Tell me the answer to this question: what color is a banana?"
input_image =  Image.open(r"sample_image.jpg")

output = MMEncoder(input_text, input_image)
print(output)