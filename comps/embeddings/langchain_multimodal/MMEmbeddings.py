from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
)
import sys
sys.path.append('../') # to load bridgetower_custom
from BridgeTowerCustom.bridgetower_custom import BridgeTowerTextFeatureExtractor, BridgeTowerForITC
from transformers import BridgeTowerProcessor
import torch
from PIL import Image
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms.functional as transform


try: 
    import habana_frameworks.torch.core as htcore
    # device = 'hpu'
    device = torch.device("hpu")
except ImportModuleError : 
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

model_name = 'BridgeTower/bridgetower-large-itm-mlm-itc'
#model_name = EMBED_MODEL
print(f"Embedding model is {model_name}")


TEXT_MODEL = BridgeTowerTextFeatureExtractor.from_pretrained(model_name).to(device)
PROCESSOR = BridgeTowerProcessor.from_pretrained(model_name)
MODEL = BridgeTowerForITC.from_pretrained(model_name).to(device)

class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """ BridgeTower embedding model """

    # TODO(tile): This does not work if batch and if texts have different length. 
    # Should implement as we did in FinetunedBridgeTowerEmbeddingChestXraypy
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        encodings = PROCESSOR.tokenizer(texts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = TEXT_MODEL(**encodings)
        embeddings = outputs.cpu().numpy().tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using BridgeTower.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
        """Embed a list of image-text pairs using BridgeTower.
        Args:
            texts: The list of texts to embed.
            images: The list of path-to-images to embed
            batch_size: the batch size to process, default to 2
        Returns:
            List of embeddings, one for each image-text pairs.
        """

        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of captions should be equal to the len of images"

        image_list = []
        text_list = []
        embeddings = []
        for path_to_img, text in zip(images, texts):
            # print(path_to_img)
            img = read_image(path_to_img, mode=ImageReadMode.RGB)
            img = transform.to_pil_image(img)
            image_list.append(img)
            text_list.append(text)
            if len(text_list) == batch_size:
                batch = PROCESSOR(image_list, text_list, return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(device)
                with torch.no_grad():
                    batch_embeddings = MODEL(**batch, output_hidden_states=True)
                for i in range(len(text_list)):
                    embeddings.append(batch_embeddings.logits[i,2,:].detach().cpu().numpy().tolist())
                image_list = []
                text_list = []
        # embedding the remaining        
        if len(text_list) > 0:
            batch = PROCESSOR(image_list, text_list, return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(device)
            with torch.no_grad():
                batch_embeddings = MODEL(**batch, output_hidden_states=True)
            for i in range(len(text_list)):
                embeddings.append(batch_embeddings.logits[i,2,:].detach().cpu().numpy().tolist())
            image_list = []
            text_list = []
        return embeddings
