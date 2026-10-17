"""OpenCLIP embedding generator for images and optional text."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """Singleton/cached embedder using OpenCLIP for feature extraction."""

    _instance: Optional["ImageEmbedder"] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.openclip_model_name
        self.pretrained = pretrained or settings.openclip_pretrained
        
        # Determine device safely
        configured_device = device or settings.embedding_device
        if configured_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(configured_device)

        logger.info(
            f"Loading OpenCLIP model '{self.model_name}' (pretrained='{self.pretrained}') on {self.device}..."
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=self.device)
            # Use visual projection
            if hasattr(self.model, "visual"):
                dummy_feat = self.model.encode_image(dummy)
                self.dimension = dummy_feat.shape[-1]
            else:
                self.dimension = 512
        logger.info(f"OpenCLIP initialized successfully. Embedding dimension: {self.dimension}")

    @classmethod
    def get_instance(cls) -> "ImageEmbedder":
        """Get or initialize singleton instance outside request loops."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate an L2-normalized float32 embedding for a single PIL image.
        
        Returns:
            np.ndarray of shape (dimension,), float32.
        """
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            # L2 normalize
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            vec = features.cpu().numpy().astype(np.float32)[0]
            # Extra safety check on norm
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

    def embed_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Generate L2-normalized float32 embeddings for a batch of PIL images.
        
        Returns:
            np.ndarray of shape (batch_size, dimension), float32.
        """
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensors)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            vecs = features.cpu().numpy().astype(np.float32)
            # Normalize each vector
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
            return vecs

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an L2-normalized float32 embedding for a text string (for hybrid search).
        
        Returns:
            np.ndarray of shape (dimension,), float32.
        """
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            vec = features.cpu().numpy().astype(np.float32)[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata for recording in index manifest."""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "dimension": self.dimension,
            "device": str(self.device),
            "open_clip_version": open_clip.__version__,
            "torch_version": torch.__version__,
        }

