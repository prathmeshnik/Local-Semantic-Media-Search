import torch
import torch.nn.functional as F
import unicodedata
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from qwen_vl_utils.vision_process import process_vision_info

# Constants for configuration
MAX_LENGTH = 8192
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 1800 * 32 * 32

@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None

class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    config: Qwen3VLConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

class Qwen3VLEmbedder:
    def __init__(self, model_name_or_path: str, device: str = "cpu", **kwargs):
        self.device = device
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(self.device).eval()
        self.processor = Qwen3VLProcessor.from_pretrained(model_name_or_path, padding_side='right')

    def _pooling_last(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @torch.no_grad()
    def embed(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        conversations = []
        for ele in inputs:
            content = []
            if ele.get('image'):
                image = ele['image']
                image_content = image if isinstance(image, Image.Image) else f"file://{image}"
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS
                })
            if ele.get('text'):
                content.append({'type': 'text', 'text': ele['text']})
            
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": content}
            ])

        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, _ = process_vision_info(conversations)
        
        processed_inputs = self.processor(
            text=text, images=images, truncation=True, 
            max_length=MAX_LENGTH, padding=True, return_tensors='pt'
        ).to(self.device)

        outputs = self.model(**processed_inputs)
        embeddings = self._pooling_last(outputs.last_hidden_state, processed_inputs['attention_mask'])

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
