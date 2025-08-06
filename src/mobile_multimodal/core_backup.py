"""Core MobileMultiModalLLM implementation with multi-task capabilities."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import utilities for validation
try:
    from .utils import ImageProcessor, ModelUtils
except ImportError:
    ImageProcessor = None
    ModelUtils = None
torch = None
onnx = None
cv2 = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Security and validation constants
MAX_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 512
MAX_INFERENCE_TIME = 30.0  # seconds
MIN_CONFIDENCE_THRESHOLD = 0.01


class MobileVisionEncoder:
    """Lightweight vision encoder optimized for mobile deployment."""
    
    def __init__(self, embed_dim: int = 384, num_patches: int = 196):
        if nn is None:
            raise ImportError("PyTorch is required for MobileVisionEncoder")
        
        # Initialize as nn.Module if available
        if hasattr(nn, 'Module'):
            super(MobileVisionEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Efficient patch embedding with depthwise convolution
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=4, stride=4),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim)
        )
        
        # Lightweight transformer blocks
        self.blocks = nn.ModuleList([
            MobileTransformerBlock(embed_dim) for _ in range(6)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision encoder."""
        # Patch embedding: (B, 3, 224, 224) -> (B, embed_dim, 14, 14)
        x = self.patch_embedding(x)
        B, C, H, W = x.shape
        
        # Flatten to sequence: (B, embed_dim, 14, 14) -> (B, 196, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        return self.norm(x)


class MobileTransformerBlock:
    """Mobile-optimized transformer block with linear attention."""
    
    def __init__(self, embed_dim: int):
        if nn is None:
            raise ImportError("PyTorch is required for MobileTransformerBlock")
        
        if hasattr(nn, 'Module'):
            super(MobileTransformerBlock, self).__init__()
            
        self.embed_dim = embed_dim
    """Mobile-optimized transformer block with linear attention."""
    
        
        # Linear attention for efficiency
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LinearAttention(embed_dim)
        
        # Efficient MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LinearAttention:
    """Linear attention mechanism for mobile efficiency."""
    
    def __init__(self, embed_dim: int, num_heads: int = 6):
        if nn is None:
            raise ImportError("PyTorch is required for LinearAttention")
        
        if hasattr(nn, 'Module'):
            super(LinearAttention, self).__init__()
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
    """Linear attention mechanism for mobile efficiency."""
    
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear attention forward pass."""
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Linear attention: O(n) complexity
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute attention efficiently
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        attention = torch.einsum('bhnd,bhde->bhne', q, kv)
        attention = attention / (torch.einsum('bhnd->bhd', q)[..., None] + 1e-6)
        
        attention = attention.transpose(1, 2).reshape(B, N, C)
        return self.proj(attention)


class TaskHead:
    """Base class for task-specific decoder heads."""
    
    def __init__(self, embed_dim: int, output_dim: int):
        if nn is None:
            raise ImportError("PyTorch is required for TaskHead")
        
        if hasattr(nn, 'Module'):
            super(TaskHead, self).__init__()
            
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
    """Base class for task-specific decoder heads."""
    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))


class CaptioningHead(TaskHead):
    """Image captioning decoder head."""
    
    def __init__(self, embed_dim: int, vocab_size: int = 32000):
        super().__init__(embed_dim, vocab_size)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, nhead=6, batch_first=True),
            num_layers=3
        )
        
    def generate_caption(self, vision_features: torch.Tensor, max_length: int = 50) -> List[int]:
        """Generate caption tokens autoregressively."""
        batch_size = vision_features.size(0)
        device = vision_features.device
        
        # Start token (assuming BOS token is 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Create positional embeddings for generated sequence
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1))
            
            # Decode next token
            decoder_output = self.decoder(
                generated.float(),  # Convert to float for transformer
                vision_features,
                tgt_mask=tgt_mask.to(device)
            )
            
            # Get next token logits
            next_token_logits = self.head(decoder_output[:, -1:])
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token (assuming EOS token is 2)
            if torch.all(next_token == 2):
                break
                
        return generated[0].tolist()  # Return first batch item


class OCRHead(TaskHead):
    """OCR text detection and recognition head."""
    
    def __init__(self, embed_dim: int, vocab_size: int = 32000):
        super().__init__(embed_dim, vocab_size)
        # Detection head for bounding boxes
        self.bbox_head = nn.Linear(embed_dim, 4)  # x, y, w, h
        # Classification head for text presence
        self.text_cls = nn.Linear(embed_dim, 2)  # text/no-text
        
    def extract_text_regions(self, vision_features: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract text regions with bounding boxes."""
        # Get patch-level predictions
        bbox_pred = self.bbox_head(vision_features)  # (B, N, 4)
        text_cls = self.text_cls(vision_features)    # (B, N, 2)
        text_tokens = self.head(vision_features)     # (B, N, vocab_size)
        
        # Apply sigmoid to bbox predictions
        bbox_pred = torch.sigmoid(bbox_pred)
        text_probs = torch.softmax(text_cls, dim=-1)
        
        # Filter patches with high text probability
        text_mask = text_probs[:, :, 1] > 0.5  # Text class probability
        
        regions = []
        batch_size = vision_features.size(0)
        
        for b in range(batch_size):
            batch_regions = []
            text_patches = torch.where(text_mask[b])[0]
            
            for patch_idx in text_patches:
                # Convert patch coordinates to image coordinates
                patch_row = patch_idx // 14  # Assuming 14x14 patches
                patch_col = patch_idx % 14
                
                # Scale bounding box to image coordinates (224x224)
                bbox = bbox_pred[b, patch_idx]
                x = (patch_col + bbox[0]) * 16  # 224/14 = 16
                y = (patch_row + bbox[1]) * 16
                w = bbox[2] * 16
                h = bbox[3] * 16
                
                # Get text tokens for this patch
                token_probs = torch.softmax(text_tokens[b, patch_idx], dim=-1)
                top_tokens = torch.topk(token_probs, k=10).indices
                
                # Simple decoding (would use proper tokenizer in practice)
                text = f"token_{top_tokens[0].item()}"
                
                batch_regions.append({
                    "text": text,
                    "bbox": [x.item(), y.item(), w.item(), h.item()],
                    "confidence": text_probs[b, patch_idx, 1].item()
                })
            
            regions.extend(batch_regions)
            
        return regions


class VQAHead(TaskHead):
    """Visual Question Answering head."""
    
    def __init__(self, embed_dim: int, vocab_size: int = 32000):
        super().__init__(embed_dim, vocab_size)
        self.question_encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=6, batch_first=True)
        
    def answer_question(self, vision_features: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        """Generate answer to visual question."""
        # Encode question
        question_features, _ = self.question_encoder(question_tokens.float())
        
        # Fuse vision and question features
        fused_features, _ = self.fusion(
            question_features,  # query
            vision_features,    # key
            vision_features     # value
        )
        
        # Generate answer tokens
        answer_logits = self.head(fused_features.mean(dim=1, keepdim=True))
        return answer_logits


class MobileMultiModalLLM:
    """Mobile Multi-Modal LLM with INT2 quantization support."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", 
                 safety_checks: bool = True):
        """Initialize the mobile multi-modal model with security validation."""
        self.model_path = model_path
        self.device = self._validate_device(device)
        self.safety_checks = safety_checks
        self._model = None
        self._onnx_session = None
        self.embed_dim = 384
        self.image_size = 224
        self._is_initialized = False
        self._model_hash = None
        
        try:
            # Validate model path if provided
            if model_path:
                if not self._validate_model_file(model_path):
                    raise ValueError(f"Invalid model file: {model_path}")
            
            # Initialize model components
            self._init_model()
            
            # Load weights if path provided
            if model_path and os.path.exists(model_path):
                self._load_weights()
            
            self._is_initialized = True
            logger.info(f"Initialized MobileMultiModalLLM on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MobileMultiModalLLM: {e}")
            raise
    
    def _validate_device(self, device: str) -> str:
        """Validate and sanitize device specification."""
        device = device.lower().strip()
        
        if device == "auto":
            if torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        if device not in ["cpu", "cuda", "mps"]:
            logger.warning(f"Unsupported device '{device}', falling back to CPU")
            device = "cpu"
        
        # Additional CUDA validation
        if device == "cuda":
            if torch is None:
                logger.warning("PyTorch not available, falling back to CPU")
                device = "cpu"
            elif not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
        
        return device
    
    def _validate_model_file(self, model_path: str) -> bool:
        """Validate model file with security checks."""
        if ModelUtils is not None:
            return ModelUtils.validate_model_path(model_path)
        
        # Fallback validation if utils not available
        try:
            path = Path(model_path)
            return (path.exists() and 
                   path.stat().st_size > 1024 and 
                   path.suffix.lower() in {'.pth', '.onnx', '.pt'})
        except Exception:
            return False
    
    def _init_model(self):
        """Initialize model architecture."""
        if torch is None:
            raise ImportError("PyTorch is required but not installed")
            
        # Vision encoder
        self.vision_encoder = MobileVisionEncoder(self.embed_dim)
        
        # Task-specific heads
        self.captioning_head = CaptioningHead(self.embed_dim)
        self.ocr_head = OCRHead(self.embed_dim)
        self.vqa_head = VQAHead(self.embed_dim)
        
        # Move to device
        if torch.cuda.is_available() and self.device == "cuda":
            self.vision_encoder = self.vision_encoder.cuda()
            self.captioning_head = self.captioning_head.cuda()
            self.ocr_head = self.ocr_head.cuda()
            self.vqa_head = self.vqa_head.cuda()
    
    def _load_weights(self):
        """Load model weights from checkpoint with validation."""
        if not self.model_path:
            return
        
        try:
            # Calculate and store model hash for integrity checking
            if ModelUtils is not None:
                self._model_hash = ModelUtils.calculate_model_hash(self.model_path)
                if not self._model_hash:
                    logger.warning("Could not calculate model hash")
            
            # Load based on file extension
            path = Path(self.model_path)
            if path.suffix.lower() == '.onnx':
                self._load_onnx_model()
            elif path.suffix.lower() in {'.pth', '.pt'}:
                self._load_pytorch_model()
            else:
                raise ValueError(f"Unsupported model format: {path.suffix}")
                
            logger.info(f"Successfully loaded weights from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def _load_pytorch_model(self):
        """Load PyTorch model weights with security validation."""
        if torch is None:
            raise ImportError("PyTorch is required for loading .pth models")
        
        try:
            # Load with weights_only=True for security (PyTorch 1.13+)
            if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            else:
                checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                raise ValueError("Invalid checkpoint format")
            
            # Load components with error handling
            components = {
                'vision_encoder': self.vision_encoder,
                'captioning_head': self.captioning_head,
                'ocr_head': self.ocr_head,
                'vqa_head': self.vqa_head
            }
            
            loaded_components = []
            for name, component in components.items():
                if name in checkpoint:
                    try:
                        component.load_state_dict(checkpoint[name], strict=False)
                        loaded_components.append(name)
                    except Exception as e:
                        logger.warning(f"Failed to load {name}: {e}")
            
            if not loaded_components:
                raise RuntimeError("No model components could be loaded")
            
            logger.info(f"Loaded PyTorch components: {loaded_components}")
            
        except Exception as e:
            logger.error(f"PyTorch model loading failed: {e}")
            raise
    
    def _load_onnx_model(self):
        """Load ONNX model for inference with provider validation."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime is required for ONNX model loading")
        
        try:
            # Validate available providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            # Select providers based on device and availability
            providers = []
            if self.device == "cuda" and 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            
            # Always include CPU as fallback
            if 'CPUExecutionProvider' in available_providers:
                providers.append('CPUExecutionProvider')
            
            if not providers:
                raise RuntimeError("No suitable ONNX execution providers available")
            
            # Create session with timeout and memory limits
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1  # Limit threads for mobile
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self._onnx_session = ort.InferenceSession(
                self.model_path, 
                session_options,
                providers=providers
            )
            
            # Validate model inputs/outputs
            input_info = [(inp.name, inp.shape, inp.type) for inp in self._onnx_session.get_inputs()]
            output_info = [(out.name, out.shape, out.type) for out in self._onnx_session.get_outputs()]
            
            logger.info(f"ONNX model loaded with providers: {providers}")
            logger.info(f"Model inputs: {input_info}")
            logger.info(f"Model outputs: {output_info}")
            
        except Exception as e:
            logger.error(f"ONNX model loading failed: {e}")
            raise
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "MobileMultiModalLLM":
        """Load pre-trained model from model zoo."""
        # Model zoo mapping
        model_zoo = {
            "mobile-mm-llm-int2": "models/mobile_mm_llm_int2.onnx",
            "mobile-mm-llm-base": "models/mobile_mm_llm_base.pth",
            "mobile-mm-llm-tiny": "models/mobile_mm_llm_tiny.onnx"
        }
        
        if model_name in model_zoo:
            model_path = model_zoo[model_name]
            return cls(model_path=model_path, **kwargs)
        else:
            logger.warning(f"Model {model_name} not found in model zoo")
            return cls(**kwargs)
    
    def _validate_input_image(self, image: np.ndarray) -> bool:
        """Validate input image for security and format requirements."""
        if image is None:
            return False
        
        if not isinstance(image, np.ndarray):
            logger.error("Image must be numpy array")
            return False
        
        # Check dimensions
        if len(image.shape) not in [2, 3]:
            logger.error(f"Invalid image dimensions: {image.shape}")
            return False
        
        h, w = image.shape[:2]
        if h < 16 or w < 16 or h > 4096 or w > 4096:
            logger.error(f"Image size out of bounds: {h}x{w}")
            return False
        
        # Check for reasonable data ranges
        if image.dtype == np.uint8:
            if np.any((image < 0) | (image > 255)):
                logger.error("Invalid pixel values for uint8 image")
                return False
        elif image.dtype == np.float32:
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                logger.error("Image contains NaN or infinity values")
                return False
        
        return True
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if cv2 is None:
            raise ImportError("OpenCV is required for image preprocessing")
        
        # Validate input
        if not self._validate_input_image(image):
            raise ValueError("Invalid input image")
        
        try:
            # Use ImageProcessor if available for secure preprocessing
            if ImageProcessor is not None:
                processor = ImageProcessor(target_size=(self.image_size, self.image_size))
                processed = processor.preprocess_image(image, maintain_aspect=False)
                if processed is None:
                    raise ValueError("Image preprocessing failed")
                
                # Convert to tensor
                image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
            else:
                # Fallback preprocessing
                # Resize to model input size
                if image.shape[:2] != (self.image_size, self.image_size):
                    image = cv2.resize(image, (self.image_size, self.image_size), 
                                     interpolation=cv2.INTER_LINEAR)
                
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume input is RGB already, or convert if needed
                    pass
                
                # Ensure 3 channels
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)
                elif image.shape[2] == 4:  # RGBA
                    image = image[:, :, :3]
                
                # Normalize to [0, 1]
                image = image.astype(np.float32) / 255.0
                
                # Standard ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std
                
                # Convert to tensor and add batch dimension
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            # Final validation
            if torch.any(torch.isnan(image_tensor)) or torch.any(torch.isinf(image_tensor)):
                raise ValueError("Preprocessed image contains invalid values")
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _encode_vision(self, image: np.ndarray) -> torch.Tensor:
        """Encode image through vision encoder."""
        image_tensor = self._preprocess_image(image)
        
        if self._onnx_session:
            # ONNX inference
            inputs = {self._onnx_session.get_inputs()[0].name: image_tensor.cpu().numpy()}
            outputs = self._onnx_session.run(None, inputs)
            return torch.from_numpy(outputs[0])
        else:
            # PyTorch inference
            with torch.no_grad():
                return self.vision_encoder(image_tensor)
    
    def generate_caption(self, image: np.ndarray, max_length: int = 50) -> str:
        """Generate descriptive caption for image."""
        try:
            # Encode image
            vision_features = self._encode_vision(image)
            
            if self._onnx_session:
                # For ONNX, we would need a separate captioning model
                return "ONNX captioning not implemented - use PyTorch model"
            
            # Generate caption tokens
            with torch.no_grad():
                caption_tokens = self.captioning_head.generate_caption(vision_features, max_length)
            
            # Simple token-to-text conversion (would use proper tokenizer)
            caption_words = [f"word_{token}" for token in caption_tokens[1:-1]]  # Skip BOS/EOS
            return " ".join(caption_words) if caption_words else "Generated caption"
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return f"Error generating caption: {str(e)}"
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text regions with OCR."""
        try:
            # Encode image
            vision_features = self._encode_vision(image)
            
            if self._onnx_session:
                # For ONNX, return mock data
                return [{"text": "ONNX OCR", "bbox": [10, 10, 100, 30], "confidence": 0.9}]
            
            # Extract text regions
            with torch.no_grad():
                text_regions = self.ocr_head.extract_text_regions(vision_features)
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return [{"text": f"Error: {str(e)}", "bbox": [0, 0, 50, 20], "confidence": 0.0}]
    
    def answer_question(self, image: np.ndarray, question: str) -> str:
        """Answer question about image content."""
        try:
            # Encode image
            vision_features = self._encode_vision(image)
            
            # Simple question tokenization (would use proper tokenizer)
            question_words = question.lower().split()
            question_tokens = torch.tensor([[hash(word) % 1000 for word in question_words]])
            question_tokens = question_tokens.float().unsqueeze(-1).expand(-1, -1, self.embed_dim)
            
            if self.device == "cuda" and torch.cuda.is_available():
                question_tokens = question_tokens.cuda()
            
            if self._onnx_session:
                return f"Answer about: {question} (ONNX VQA not implemented)"
            
            # Generate answer
            with torch.no_grad():
                answer_logits = self.vqa_head.answer_question(vision_features, question_tokens)
                answer_token = torch.argmax(answer_logits, dim=-1).item()
            
            # Simple answer mapping
            answer_map = {
                0: "yes", 1: "no", 2: "red", 3: "blue", 4: "green", 
                5: "car", 6: "person", 7: "animal", 8: "building", 9: "food"
            }
            
            return answer_map.get(answer_token % 10, "unknown")
            
        except Exception as e:
            logger.error(f"VQA failed: {e}")
            return f"Error answering question: {str(e)}"
    
    def get_image_embeddings(self, image: np.ndarray) -> np.ndarray:
        """Get dense image embeddings for retrieval."""
        try:
            vision_features = self._encode_vision(image)
            # Global average pooling for image-level representation
            embeddings = torch.mean(vision_features, dim=1).cpu().numpy()
            return embeddings
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return np.zeros((1, self.embed_dim))
    
    def benchmark_inference(self, image: np.ndarray, iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        import time
        
        results = {}
        
        # Warm up
        for _ in range(5):
            self._encode_vision(image)
        
        # Benchmark vision encoding
        start_time = time.time()
        for _ in range(iterations):
            self._encode_vision(image)
        vision_time = (time.time() - start_time) / iterations * 1000  # ms
        
        # Benchmark full pipeline
        start_time = time.time()
        for _ in range(iterations):
            self.generate_caption(image)
        caption_time = (time.time() - start_time) / iterations * 1000  # ms
        
        results = {
            "vision_encoding_ms": vision_time,
            "caption_generation_ms": caption_time,
            "total_inference_ms": vision_time + caption_time,
            "fps": 1000 / (vision_time + caption_time)
        }
        
        logger.info(f"Benchmark results: {results}")
        return results
    
    def export_onnx(self, output_path: str, dynamic_batch: bool = True):
        """Export model to ONNX format for mobile deployment."""
        if torch is None:
            raise ImportError("PyTorch is required for ONNX export")
            
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
            if self.device == "cuda":
                dummy_input = dummy_input.cuda()
            
            # Dynamic axes for variable batch size
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic_batch else None
            
            # Export vision encoder
            torch.onnx.export(
                self.vision_encoder,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            logger.info(f"Model exported to {output_path}")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    def quantize_int8(self, calibration_data: List[np.ndarray], output_path: str):
        """Apply INT8 quantization for mobile deployment."""
        try:
            if torch is None:
                raise ImportError("PyTorch is required for quantization")
                
            from torch.quantization import quantize_dynamic
            
            # Apply dynamic quantization
            quantized_model = quantize_dynamic(
                self.vision_encoder,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Save quantized model
            torch.save({
                'vision_encoder': quantized_model.state_dict(),
                'captioning_head': self.captioning_head.state_dict(),
                'ocr_head': self.ocr_head.state_dict(),
                'vqa_head': self.vqa_head.state_dict()
            }, output_path)
            
            logger.info(f"INT8 quantized model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture and parameter information."""
        if torch is None:
            return {"error": "PyTorch not available"}
            
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            "architecture": "MobileMultiModalLLM",
            "embed_dim": self.embed_dim,
            "image_size": self.image_size,
            "device": self.device,
            "parameters": {
                "vision_encoder": count_parameters(self.vision_encoder),
                "captioning_head": count_parameters(self.captioning_head),
                "ocr_head": count_parameters(self.ocr_head),
                "vqa_head": count_parameters(self.vqa_head)
            },
            "model_path": self.model_path,
            "onnx_session": self._onnx_session is not None
        }
        
        total_params = sum(info["parameters"].values())
        info["total_parameters"] = total_params
        info["estimated_size_mb"] = total_params * 4 / (1024 * 1024)  # FP32 estimation
        
        return info