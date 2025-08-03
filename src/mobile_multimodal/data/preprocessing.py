"""Data preprocessing pipeline for mobile multi-modal models."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    import torchvision.transforms as T
    from torchvision.transforms import functional as TF
except ImportError:
    torch = None
    T = None
    TF = None

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Base class for data preprocessing pipelines."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or {}
        self.transforms = []
        self.statistics = {}
    
    def add_transform(self, transform: Callable, **kwargs):
        """Add a transformation to the pipeline."""
        self.transforms.append((transform, kwargs))
    
    def process(self, data: Any) -> Any:
        """Apply all transformations to data."""
        for transform, kwargs in self.transforms:
            data = transform(data, **kwargs)
        return data
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a batch of data."""
        return [self.process(item) for item in batch]
    
    def compute_statistics(self, dataset: List[Any]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        logger.info("Computing dataset statistics...")
        
        self.statistics = {
            "sample_count": len(dataset),
            "computed_at": str(time.time()) if 'time' in globals() else "unknown"
        }
        
        return self.statistics
    
    def save_config(self, path: str):
        """Save preprocessing configuration."""
        config_data = {
            "config": self.config,
            "statistics": self.statistics,
            "transform_count": len(self.transforms)
        }
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, path: str):
        """Load preprocessing configuration."""
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        self.config = config_data.get("config", {})
        self.statistics = config_data.get("statistics", {})


class ImagePreprocessor(DataPreprocessor):
    """Image preprocessing pipeline for mobile models."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = False,
                 **kwargs):
        """Initialize image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to apply ImageNet normalization
            augment: Whether to apply data augmentation
        """
        super().__init__(kwargs)
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        
        # ImageNet statistics
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transformation pipeline."""
        # Basic transforms
        self.add_transform(self._resize_image)
        
        if self.augment:
            self.add_transform(self._augment_image)
        
        if self.normalize:
            self.add_transform(self._normalize_image)
        
        self.add_transform(self._to_tensor)
    
    def _resize_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Resize image to target size."""
        if cv2 is None:
            raise ImportError("OpenCV required for image resizing")
        
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def _augment_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply data augmentation."""
        if cv2 is None:
            return image
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return image
    
    def _normalize_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply ImageNet normalization."""
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - self.mean) / self.std
        
        return image
    
    def _to_tensor(self, image: np.ndarray, **kwargs) -> Union[np.ndarray, 'torch.Tensor']:
        """Convert to tensor format."""
        if torch is not None:
            # Convert to PyTorch tensor
            if len(image.shape) == 3:
                # HWC to CHW
                image = np.transpose(image, (2, 0, 1))
            return torch.from_numpy(image.copy()).float()
        else:
            # Keep as numpy array
            if len(image.shape) == 3:
                # HWC to CHW for consistency
                image = np.transpose(image, (2, 0, 1))
            return image.astype(np.float32)
    
    def compute_statistics(self, dataset: List[np.ndarray]) -> Dict[str, Any]:
        """Compute image dataset statistics."""
        super().compute_statistics(dataset)
        
        if not dataset:
            return self.statistics
        
        # Sample subset for statistics
        sample_size = min(1000, len(dataset))
        sample_indices = random.sample(range(len(dataset)), sample_size)
        
        # Compute pixel statistics
        pixel_values = []
        image_sizes = []
        
        for idx in sample_indices:
            image = dataset[idx]
            if isinstance(image, str):
                # Load image if path provided
                image = cv2.imread(image)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is not None:
                image_sizes.append(image.shape[:2])
                pixel_values.extend(image.flatten())
        
        if pixel_values:
            pixel_values = np.array(pixel_values, dtype=np.float32)
            
            self.statistics.update({
                "pixel_mean": float(np.mean(pixel_values)),
                "pixel_std": float(np.std(pixel_values)),
                "pixel_min": float(np.min(pixel_values)),
                "pixel_max": float(np.max(pixel_values)),
                "common_sizes": self._get_common_sizes(image_sizes),
                "target_size": self.target_size
            })
        
        return self.statistics
    
    def _get_common_sizes(self, sizes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get most common image sizes."""
        size_counts = {}
        for size in sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Return top 5 most common sizes
        sorted_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
        return [size for size, count in sorted_sizes[:5]]


class TextPreprocessor(DataPreprocessor):
    """Text preprocessing pipeline for captions and questions."""
    
    def __init__(self,
                 max_length: int = 50,
                 vocab_size: int = 32000,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 **kwargs):
        """Initialize text preprocessor.
        
        Args:
            max_length: Maximum sequence length
            vocab_size: Vocabulary size
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
        """
        super().__init__(kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
        # Vocabulary
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            "[PAD]": 0,
            "[BOS]": 1, 
            "[EOS]": 2,
            "[UNK]": 3
        }
        
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup text transformation pipeline."""
        self.add_transform(self._clean_text)
        self.add_transform(self._tokenize_text)
        self.add_transform(self._encode_tokens)
        self.add_transform(self._pad_sequence)
    
    def _clean_text(self, text: str, **kwargs) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            import re
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize_text(self, text: str, **kwargs) -> List[str]:
        """Tokenize text into words."""
        return text.split()
    
    def _encode_tokens(self, tokens: List[str], **kwargs) -> List[int]:
        """Encode tokens to IDs."""
        encoded = [self.special_tokens["[BOS]"]]
        
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                encoded.append(self.special_tokens["[UNK]"])
        
        encoded.append(self.special_tokens["[EOS]"])
        
        return encoded
    
    def _pad_sequence(self, sequence: List[int], **kwargs) -> List[int]:
        """Pad or truncate sequence to max length."""
        if len(sequence) > self.max_length:
            # Truncate but keep EOS token
            sequence = sequence[:self.max_length-1] + [self.special_tokens["[EOS]"]]
        elif len(sequence) < self.max_length:
            # Pad with PAD tokens
            sequence.extend([self.special_tokens["[PAD]"]] * (self.max_length - len(sequence)))
        
        return sequence
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from text corpus."""
        # Count word frequencies
        word_freq = {}
        for text in texts:
            tokens = self._tokenize_text(self._clean_text(text))
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        self.vocab = self.special_tokens.copy()
        next_id = len(self.special_tokens)
        
        for word, freq in sorted_words:
            if freq >= min_freq and next_id < self.vocab_size:
                self.vocab[word] = next_id
                next_id += 1
        
        # Build inverse vocabulary
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")
        return self.vocab
    
    def decode_sequence(self, sequence: List[int]) -> str:
        """Decode sequence of IDs back to text."""
        words = []
        for token_id in sequence:
            if token_id in self.inverse_vocab:
                word = self.inverse_vocab[token_id]
                if word not in ["[PAD]", "[BOS]", "[EOS]"]:
                    words.append(word)
        
        return " ".join(words)
    
    def compute_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Compute text dataset statistics."""
        super().compute_statistics(texts)
        
        if not texts:
            return self.statistics
        
        # Compute text statistics
        lengths = []
        word_counts = {}
        char_counts = {}
        
        for text in texts:
            clean_text = self._clean_text(text)
            tokens = self._tokenize_text(clean_text)
            
            lengths.append(len(tokens))
            
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            
            for char in clean_text:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        self.statistics.update({
            "text_count": len(texts),
            "avg_length": float(np.mean(lengths)),
            "max_length": int(np.max(lengths)),
            "min_length": int(np.min(lengths)),
            "std_length": float(np.std(lengths)),
            "vocab_size": len(word_counts),
            "unique_chars": len(char_counts),
            "most_common_words": sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        })
        
        return self.statistics


class AugmentationPipeline:
    """Advanced augmentation pipeline for multi-modal data."""
    
    def __init__(self, 
                 image_aug_prob: float = 0.5,
                 text_aug_prob: float = 0.3,
                 mix_up_prob: float = 0.1):
        """Initialize augmentation pipeline.
        
        Args:
            image_aug_prob: Probability of applying image augmentation
            text_aug_prob: Probability of applying text augmentation  
            mix_up_prob: Probability of applying mixup augmentation
        """
        self.image_aug_prob = image_aug_prob
        self.text_aug_prob = text_aug_prob
        self.mix_up_prob = mix_up_prob
        
        self.image_augmentations = [
            self._color_jitter,
            self._random_blur,
            self._random_noise,
            self._cutout
        ]
        
        self.text_augmentations = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_swap,
            self._random_deletion
        ]
    
    def augment_sample(self, image: np.ndarray, text: str) -> Tuple[np.ndarray, str]:
        """Apply augmentation to a single sample."""
        # Image augmentation
        if random.random() < self.image_aug_prob:
            aug_func = random.choice(self.image_augmentations)
            image = aug_func(image)
        
        # Text augmentation
        if random.random() < self.text_aug_prob:
            aug_func = random.choice(self.text_augmentations)
            text = aug_func(text)
        
        return image, text
    
    def augment_batch(self, batch: List[Tuple[np.ndarray, str]]) -> List[Tuple[np.ndarray, str]]:
        """Apply augmentation to a batch with potential mixup."""
        augmented_batch = []
        
        for image, text in batch:
            # Apply individual augmentations
            aug_image, aug_text = self.augment_sample(image, text)
            augmented_batch.append((aug_image, aug_text))
        
        # Apply mixup
        if random.random() < self.mix_up_prob and len(augmented_batch) > 1:
            augmented_batch = self._apply_mixup(augmented_batch)
        
        return augmented_batch
    
    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jittering."""
        if cv2 is None:
            return image
        
        # Random brightness
        alpha = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        # Random saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image
    
    def _random_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur."""
        if cv2 is None:
            return image
        
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _random_noise(self, image: np.ndarray) -> np.ndarray:
        """Add random noise."""
        noise = np.random.normal(0, 25, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def _cutout(self, image: np.ndarray) -> np.ndarray:
        """Apply cutout augmentation."""
        h, w = image.shape[:2]
        
        # Random cutout size (5-20% of image)
        cut_size = random.randint(int(0.05 * min(h, w)), int(0.2 * min(h, w)))
        
        # Random position
        x = random.randint(0, w - cut_size)
        y = random.randint(0, h - cut_size)
        
        # Apply cutout
        image_copy = image.copy()
        image_copy[y:y+cut_size, x:x+cut_size] = 0
        
        return image_copy
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms (simplified)."""
        words = text.split()
        if not words:
            return text
        
        # Simple synonym replacement (in practice, use WordNet or word embeddings)
        synonyms = {
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"]
        }
        
        # Replace random word
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        if word.lower() in synonyms:
            words[word_idx] = random.choice(synonyms[word.lower()])
        
        return " ".join(words)
    
    def _random_insertion(self, text: str) -> str:
        """Insert random words."""
        words = text.split()
        if not words:
            return text
        
        # Insert common words randomly
        common_words = ["the", "a", "an", "very", "quite", "really"]
        insert_word = random.choice(common_words)
        insert_pos = random.randint(0, len(words))
        
        words.insert(insert_pos, insert_word)
        return " ".join(words)
    
    def _random_swap(self, text: str) -> str:
        """Swap random words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return " ".join(words)
    
    def _random_deletion(self, text: str) -> str:
        """Delete random words."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Delete up to 20% of words
        num_to_delete = random.randint(1, max(1, len(words) // 5))
        indices_to_delete = random.sample(range(len(words)), num_to_delete)
        
        for idx in sorted(indices_to_delete, reverse=True):
            del words[idx]
        
        return " ".join(words)
    
    def _apply_mixup(self, batch: List[Tuple[np.ndarray, str]], alpha: float = 0.2) -> List[Tuple[np.ndarray, str]]:
        """Apply mixup augmentation to batch."""
        if len(batch) < 2:
            return batch
        
        mixed_batch = []
        batch_size = len(batch)
        
        for i in range(batch_size):
            # Get random pair
            j = random.randint(0, batch_size - 1)
            if i == j:
                j = (j + 1) % batch_size
            
            image1, text1 = batch[i]
            image2, text2 = batch[j]
            
            # Mix images
            lambda_val = np.random.beta(alpha, alpha)
            mixed_image = lambda_val * image1 + (1 - lambda_val) * image2
            mixed_image = mixed_image.astype(np.uint8)
            
            # For text, choose one or concatenate
            if random.random() > 0.5:
                mixed_text = text1
            else:
                mixed_text = f"{text1} {text2}"
            
            mixed_batch.append((mixed_image, mixed_text))
        
        return mixed_batch


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Testing preprocessing modules...")
    
    # Test image preprocessor
    print("Testing ImagePreprocessor...")
    
    # Create dummy images
    dummy_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8),
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    ]
    
    image_processor = ImagePreprocessor(target_size=(224, 224), augment=True)
    
    # Process single image
    processed_image = image_processor.process(dummy_images[0])
    print(f"Processed image shape: {processed_image.shape}")
    
    # Compute statistics
    stats = image_processor.compute_statistics(dummy_images)
    print(f"Image statistics: {stats}")
    
    print("✓ ImagePreprocessor works")
    
    # Test text preprocessor
    print("\nTesting TextPreprocessor...")
    
    dummy_texts = [
        "A red car driving on the road",
        "Blue sky with white clouds", 
        "Green trees in the park",
        "Yellow flowers in the garden"
    ]
    
    text_processor = TextPreprocessor(max_length=20)
    
    # Build vocabulary
    vocab = text_processor.build_vocabulary(dummy_texts)
    print(f"Built vocabulary with {len(vocab)} tokens")
    
    # Process text
    processed_text = text_processor.process(dummy_texts[0])
    print(f"Processed text: {processed_text}")
    
    # Decode back
    decoded = text_processor.decode_sequence(processed_text)
    print(f"Decoded text: {decoded}")
    
    # Compute statistics
    text_stats = text_processor.compute_statistics(dummy_texts)
    print(f"Text statistics: {text_stats}")
    
    print("✓ TextPreprocessor works")
    
    # Test augmentation pipeline
    print("\nTesting AugmentationPipeline...")
    
    augmentation = AugmentationPipeline()
    
    # Test single sample augmentation
    test_image = dummy_images[0]
    test_text = dummy_texts[0]
    
    aug_image, aug_text = augmentation.augment_sample(test_image, test_text)
    print(f"Original text: {test_text}")
    print(f"Augmented text: {aug_text}")
    
    # Test batch augmentation
    test_batch = [(dummy_images[i], dummy_texts[i]) for i in range(len(dummy_images))]
    aug_batch = augmentation.augment_batch(test_batch)
    print(f"Augmented batch size: {len(aug_batch)}")
    
    print("✓ AugmentationPipeline works")
    
    print("\nAll preprocessing tests passed!")