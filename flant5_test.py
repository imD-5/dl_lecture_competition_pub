from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import torch

"""
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", device_map="auto")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

input_text = "answer this: what ae you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
from transformers import ViTMAEModel, AutoImageProcessor
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
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", device_map="auto")
        self.text_Encoder = T5EncoderModel.from_pretrained("google/flan-t5-base", device_map="auto")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto").to(self.device)

        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(self.device)
        
        self.cross_attention = CrossAttention(768).to(self.device)
        
    def forward(self, text_input, image_input):

        text_tokens = self.tokenizer(text_input, return_tensors="pt").input_ids.to("cuda")
        encoded_text = self.text_Encoder(text_tokens).last_hidden_state
        
        image_inputs = self.image_transforms(image_input).unsqueeze(0).to(self.device)
        encoded_image = self.image_encoder(image_inputs).last_hidden_state
        
        print(f"Encoded text shape: {encoded_text}")
        print(f"Encoded image shape: {encoded_image}")
        
        cross_attended_text = self.cross_attention(encoded_text, encoded_image)
        combined_features = torch.cat([encoded_text, cross_attended_text], dim=1)

        decoder_outputs = self.t5_model.decoder(input_ids=encoded_text,
            attention_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        sequence_output = decoder_outputs[0]
        lm_logits = self.t5_model.lm_head(sequence_output)

        return lm_logits
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MMEncoder =  MultimodalVAE(device)
input_text = "Tell me the answer to this question: what color is a banana?"
input_image =  Image.open(r"C:\Users\kimura.daigo\Downloads\sample_image.jpg")

output = MMEncoder(input_text, input_image)
print(output)