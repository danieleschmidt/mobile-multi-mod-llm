"""Data fixtures for Mobile Multi-Modal LLM testing."""

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MockMultiModalDataset(Dataset):
    """Mock dataset for testing multimodal functionality."""
    
    def __init__(self, size: int = 100, image_size: Tuple[int, int] = (224, 224)):
        self.size = size
        self.image_size = image_size
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate mock dataset samples."""
        samples = []
        for i in range(self.size):
            sample = {
                "image_id": f"img_{i:05d}",
                "image": self._create_random_image(),
                "caption": f"This is a sample caption for image {i}",
                "ocr_text": [
                    {"text": f"Sample text {i}", "bbox": [10, 10, 100, 30]},
                    {"text": f"More text {i}", "bbox": [10, 40, 120, 60]}
                ],
                "vqa_pairs": [
                    {"question": f"What color is the object in image {i}?", "answer": "blue"},
                    {"question": f"How many objects in image {i}?", "answer": "three"}
                ],
                "metadata": {
                    "source": "mock_dataset",
                    "split": "train" if i < 80 else "val",
                    "quality_score": np.random.uniform(0.7, 1.0)
                }
            }
            samples.append(sample)
        return samples
    
    def _create_random_image(self) -> Image.Image:
        """Create a random RGB image."""
        array = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        return Image.fromarray(array)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class DataFixture:
    """Fixture class for data-related test utilities."""
    
    @staticmethod
    def create_test_dataset(size: int = 100, image_size: Tuple[int, int] = (224, 224)) -> MockMultiModalDataset:
        """Create a test dataset."""
        return MockMultiModalDataset(size, image_size)
    
    @staticmethod
    def create_sample_images(count: int = 10, size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
        """Create sample images for testing."""
        images = []
        for i in range(count):
            # Create different types of test images
            if i % 3 == 0:
                # Solid color image
                color = np.random.randint(0, 255, 3)
                array = np.full((*size, 3), color, dtype=np.uint8)
            elif i % 3 == 1:
                # Gradient image
                array = np.zeros((*size, 3), dtype=np.uint8)
                for j in range(3):
                    array[:, :, j] = np.linspace(0, 255, size[0] * size[1]).reshape(size)
            else:
                # Random noise image
                array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            
            images.append(Image.fromarray(array))
        return images
    
    @staticmethod
    def create_sample_captions(count: int = 10) -> List[str]:
        """Create sample captions for testing."""
        templates = [
            "A photo of {}",
            "An image showing {}",
            "This picture contains {}",
            "You can see {} in this image",
            "The image depicts {}"
        ]
        objects = ["a cat", "a dog", "a car", "a house", "a tree", "people", "food", "a bird", "flowers", "a building"]
        
        captions = []
        for i in range(count):
            template = templates[i % len(templates)]
            obj = objects[i % len(objects)]
            captions.append(template.format(obj))
        return captions
    
    @staticmethod
    def create_vqa_pairs(count: int = 20) -> List[Dict[str, str]]:
        """Create Visual Question-Answer pairs for testing."""
        questions = [
            "What color is the main object?",
            "How many objects are in the image?",
            "What is the weather like?",
            "Is this indoors or outdoors?",
            "What time of day is it?",
            "What activity is taking place?",
            "What emotion is shown?",
            "What material is the object made of?",
            "What is the approximate age?",
            "What season is depicted?"
        ]
        
        answers = [
            "blue", "red", "green", "yellow", "black", "white",
            "one", "two", "three", "many", "none",
            "sunny", "rainy", "cloudy", "snowy",
            "indoors", "outdoors",
            "morning", "afternoon", "evening", "night",
            "playing", "working", "sleeping", "eating",
            "happy", "sad", "angry", "surprised",
            "wood", "metal", "plastic", "fabric",
            "young", "old", "middle-aged",
            "spring", "summer", "fall", "winter"
        ]
        
        pairs = []
        for i in range(count):
            question = questions[i % len(questions)]
            answer = answers[i % len(answers)]
            pairs.append({"question": question, "answer": answer})
        
        return pairs
    
    @staticmethod
    def create_ocr_annotations(count: int = 10) -> List[List[Dict[str, Any]]]:
        """Create OCR annotations for testing."""
        annotations = []
        for i in range(count):
            # Random number of text regions per image
            num_regions = np.random.randint(1, 5)
            regions = []
            
            for j in range(num_regions):
                # Random bounding box
                x1, y1 = np.random.randint(0, 150, 2)
                x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(20, 40)
                
                region = {
                    "text": f"Sample text {i}_{j}",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": np.random.uniform(0.8, 1.0)
                }
                regions.append(region)
            
            annotations.append(regions)
        
        return annotations
    
    @staticmethod
    def create_benchmark_data() -> Dict[str, Any]:
        """Create data for benchmark testing."""
        return {
            "images": DataFixture.create_sample_images(100),
            "captions": DataFixture.create_sample_captions(100),
            "vqa_pairs": DataFixture.create_vqa_pairs(200),
            "ocr_annotations": DataFixture.create_ocr_annotations(100),
            "metadata": {
                "dataset_size": 100,
                "image_resolution": (224, 224),
                "tasks": ["captioning", "ocr", "vqa", "retrieval"],
                "quality_threshold": 0.8
            }
        }
    
    @staticmethod
    def save_test_dataset(dataset: MockMultiModalDataset, output_dir: Path) -> None:
        """Save test dataset to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "size": len(dataset),
            "image_size": dataset.image_size,
            "tasks": ["captioning", "ocr", "vqa"],
            "created_by": "test_fixture"
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save sample annotations
        annotations = []
        for i, sample in enumerate(dataset.samples[:10]):  # Save first 10 samples
            annotation = {
                "image_id": sample["image_id"],
                "caption": sample["caption"],
                "ocr_text": sample["ocr_text"],
                "vqa_pairs": sample["vqa_pairs"]
            }
            annotations.append(annotation)
            
            # Save sample image
            sample["image"].save(output_dir / f"{sample['image_id']}.jpg")
        
        with open(output_dir / "annotations.json", "w") as f:
            json.dump(annotations, f, indent=2)


def create_test_dataset(size: int = 100) -> MockMultiModalDataset:
    """Factory function to create a test dataset."""
    return DataFixture.create_test_dataset(size)


def mock_data_loader(batch_size: int = 32, num_batches: int = 10):
    """Mock data loader for testing."""
    dataset = create_test_dataset(batch_size * num_batches)
    
    def batch_generator():
        for i in range(0, len(dataset), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(dataset))):
                sample = dataset[j]
                # Convert to tensor format
                batch_sample = {
                    "image": torch.randn(3, 224, 224),  # Mock tensor
                    "caption": sample["caption"],
                    "image_id": sample["image_id"]
                }
                batch.append(batch_sample)
            yield batch
    
    return batch_generator()


def create_adversarial_samples() -> Dict[str, Any]:
    """Create adversarial samples for security testing."""
    return {
        "clean_image": np.random.rand(224, 224, 3).astype(np.float32),
        "adversarial_image": np.random.rand(224, 224, 3).astype(np.float32) + 0.01,
        "noisy_image": np.random.rand(224, 224, 3).astype(np.float32) * 255,
        "corrupted_image": np.full((224, 224, 3), np.nan),
        "oversized_image": np.random.rand(4096, 4096, 3).astype(np.float32),
        "malicious_caption": "' OR 1=1 --",
        "long_caption": "A" * 10000,
        "unicode_caption": "测试中文字符 🤖 émojis",
        "empty_inputs": {
            "image": np.array([]),
            "text": "",
            "caption": None
        }
    }