"""Dataset implementations for multi-modal training and evaluation."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = object

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """Base multi-modal dataset for training mobile models."""
    
    def __init__(self, 
                 data_path: str,
                 annotations_file: str,
                 image_size: Tuple[int, int] = (224, 224),
                 max_caption_length: int = 50,
                 max_question_length: int = 20,
                 transform=None,
                 cache_images: bool = False):
        """Initialize multi-modal dataset.
        
        Args:
            data_path: Path to dataset root directory
            annotations_file: Path to annotations JSON file
            image_size: Target image size (height, width)
            max_caption_length: Maximum caption length in tokens
            max_question_length: Maximum question length in tokens
            transform: Optional transform function
            cache_images: Whether to cache images in memory
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self.max_question_length = max_question_length
        self.transform = transform
        self.cache_images = cache_images
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        self.image_cache = {} if cache_images else None
        
        logger.info(f"Loaded {len(self.annotations)} samples from {annotations_file}")
    
    def _load_annotations(self, annotations_file: str) -> List[Dict[str, Any]]:
        """Load dataset annotations from JSON file."""
        try:
            with open(annotations_file, 'r') as f:
                data = json.load(f)
            
            # Handle different annotation formats
            if isinstance(data, list):
                return data
            elif 'annotations' in data:
                return data['annotations']
            elif 'images' in data:
                # COCO-style format
                return self._convert_coco_format(data)
            else:
                raise ValueError("Unknown annotation format")
                
        except Exception as e:
            logger.error(f"Failed to load annotations from {annotations_file}: {e}")
            return []
    
    def _convert_coco_format(self, coco_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert COCO format to our internal format."""
        annotations = []
        
        # Create image lookup
        images = {img['id']: img for img in coco_data.get('images', [])}
        
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id in images:
                image_info = images[image_id]
                
                sample = {
                    'image_id': image_id,
                    'image_path': image_info['file_name'],
                    'caption': ann.get('caption', ''),
                    'bbox': ann.get('bbox', []),
                    'category_id': ann.get('category_id', 0),
                    'area': ann.get('area', 0),
                    'iscrowd': ann.get('iscrowd', 0)
                }
                annotations.append(sample)
        
        return annotations
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item by index."""
        try:
            annotation = self.annotations[idx]
            
            # Load image
            image = self._load_image(annotation['image_path'])
            if image is None:
                # Return dummy data on image load failure
                return self._get_dummy_sample()
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Prepare sample
            sample = {
                'image': image,
                'image_id': annotation.get('image_id', idx),
                'caption': annotation.get('caption', ''),
                'bbox': annotation.get('bbox', []),
                'category_id': annotation.get('category_id', 0),
                'text_regions': annotation.get('text_regions', []),
                'questions': annotation.get('questions', []),
                'answers': annotation.get('answers', [])
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from path with caching support."""
        full_path = self.data_path / image_path
        
        # Check cache first
        if self.cache_images and str(full_path) in self.image_cache:
            return self.image_cache[str(full_path)]
        
        # Load image
        if cv2 is None:
            logger.error("OpenCV not available for image loading")
            return None
            
        try:
            image = cv2.imread(str(full_path))
            if image is None:
                logger.warning(f"Failed to load image: {full_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # Cache if enabled
            if self.cache_images:
                self.image_cache[str(full_path)] = image
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            return None
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Return dummy sample for error cases."""
        return {
            'image': np.zeros((*self.image_size, 3), dtype=np.uint8),
            'image_id': -1,
            'caption': '',
            'bbox': [],
            'category_id': 0,
            'text_regions': [],
            'questions': [],
            'answers': []
        }


class CaptioningDataset(MultiModalDataset):
    """Dataset specifically for image captioning tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = 'captioning'
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get captioning sample."""
        sample = super().__getitem__(idx)
        
        # Focus on captioning-specific data
        return {
            'image': sample['image'],
            'caption': sample['caption'],
            'image_id': sample['image_id']
        }


class OCRDataset(MultiModalDataset):
    """Dataset for OCR (Optical Character Recognition) tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = 'ocr'
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get OCR sample."""
        sample = super().__getitem__(idx)
        
        # Process text regions for OCR
        text_regions = sample.get('text_regions', [])
        if not text_regions and sample.get('caption'):
            # Create synthetic text region if only caption available
            text_regions = [{
                'text': sample['caption'],
                'bbox': [0, 0, self.image_size[1], self.image_size[0]],
                'confidence': 1.0
            }]
        
        return {
            'image': sample['image'],
            'text_regions': text_regions,
            'bbox': sample.get('bbox', []),
            'image_id': sample['image_id']
        }


class VQADataset(MultiModalDataset):
    """Dataset for Visual Question Answering tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = 'vqa'
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get VQA sample."""
        sample = super().__getitem__(idx)
        
        # Process questions and answers
        questions = sample.get('questions', [])
        answers = sample.get('answers', [])
        
        # If no questions/answers, create synthetic ones
        if not questions and sample.get('caption'):
            questions = [f"What is in this image?"]
            answers = [sample['caption']]
        
        # Ensure questions and answers are paired
        min_len = min(len(questions), len(answers)) if answers else len(questions)
        questions = questions[:min_len]
        answers = answers[:min_len] if answers else ['unknown'] * min_len
        
        return {
            'image': sample['image'],
            'questions': questions,
            'answers': answers,
            'image_id': sample['image_id']
        }


class DatasetSplitter:
    """Utility class for splitting datasets."""
    
    @staticmethod
    def train_val_test_split(dataset: Dataset, 
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.2,
                           test_ratio: float = 0.1,
                           random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train, validation, and test sets."""
        if torch is None:
            raise ImportError("PyTorch required for dataset splitting")
            
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(random_seed)
        
        return torch.utils.data.random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
    
    @staticmethod
    def stratified_split(dataset: MultiModalDataset,
                        split_ratios: List[float] = [0.7, 0.2, 0.1],
                        stratify_key: str = 'category_id') -> List[Dataset]:
        """Stratified split based on a specific key."""
        if torch is None:
            raise ImportError("PyTorch required for dataset splitting")
            
        # Group samples by stratify key
        groups = {}
        for idx, annotation in enumerate(dataset.annotations):
            key = annotation.get(stratify_key, 0)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        # Split each group proportionally
        split_indices = [[] for _ in split_ratios]
        
        for group_indices in groups.values():
            group_size = len(group_indices)
            split_sizes = [int(ratio * group_size) for ratio in split_ratios]
            
            # Adjust for rounding errors
            diff = group_size - sum(split_sizes)
            if diff > 0:
                split_sizes[0] += diff
            
            # Split this group
            start_idx = 0
            for i, size in enumerate(split_sizes):
                end_idx = start_idx + size
                split_indices[i].extend(group_indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Create subset datasets
        splits = []
        for indices in split_indices:
            subset = torch.utils.data.Subset(dataset, indices)
            splits.append(subset)
        
        return splits


def create_dataloader(dataset: Dataset,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     drop_last: bool = False,
                     collate_fn=None) -> DataLoader:
    """Create DataLoader with optimized settings for mobile training."""
    if torch is None:
        raise ImportError("PyTorch required for DataLoader creation")
    
    # Default collate function for multi-modal data
    if collate_fn is None:
        collate_fn = multimodal_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for multi-modal batches."""
    if torch is None:
        raise ImportError("PyTorch required for collate function")
    
    # Separate different data types
    images = []
    captions = []
    questions = []
    answers = []
    text_regions = []
    bboxes = []
    image_ids = []
    
    for sample in batch:
        images.append(sample.get('image'))
        captions.append(sample.get('caption', ''))
        questions.append(sample.get('questions', []))
        answers.append(sample.get('answers', []))
        text_regions.append(sample.get('text_regions', []))
        bboxes.append(sample.get('bbox', []))
        image_ids.append(sample.get('image_id', -1))
    
    # Stack images
    if images and images[0] is not None:
        # Convert to tensor and normalize
        image_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Convert numpy to tensor
                img_tensor = torch.from_numpy(img).float()
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
                # Normalize to [0, 1]
                img_tensor = img_tensor / 255.0
                image_tensors.append(img_tensor)
            else:
                image_tensors.append(img)
        
        images_batch = torch.stack(image_tensors, dim=0)
    else:
        images_batch = None
    
    return {
        'images': images_batch,
        'captions': captions,
        'questions': questions, 
        'answers': answers,
        'text_regions': text_regions,
        'bboxes': bboxes,
        'image_ids': image_ids
    }


class DatasetBuilder:
    """Builder class for creating datasets from various sources."""
    
    @staticmethod
    def from_coco_format(data_path: str, 
                        annotation_file: str,
                        task: str = 'captioning',
                        **kwargs) -> MultiModalDataset:
        """Create dataset from COCO format annotations."""
        if task == 'captioning':
            return CaptioningDataset(data_path, annotation_file, **kwargs)
        elif task == 'ocr':
            return OCRDataset(data_path, annotation_file, **kwargs)
        elif task == 'vqa':
            return VQADataset(data_path, annotation_file, **kwargs)
        else:
            return MultiModalDataset(data_path, annotation_file, **kwargs)
    
    @staticmethod
    def from_directory(data_path: str,
                      image_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
                      annotation_pattern: str = '*.json',
                      **kwargs) -> MultiModalDataset:
        """Create dataset by scanning directory structure."""
        data_path = Path(data_path)
        
        # Find all images
        images = []
        for ext in image_extensions:
            images.extend(data_path.glob(f'**/*{ext}'))
        
        # Find annotation files
        annotation_files = list(data_path.glob(annotation_pattern))
        
        if annotation_files:
            # Use first annotation file found
            return MultiModalDataset(str(data_path), str(annotation_files[0]), **kwargs)
        else:
            # Create synthetic annotations
            annotations = []
            for i, img_path in enumerate(images):
                annotations.append({
                    'image_id': i,
                    'image_path': str(img_path.relative_to(data_path)),
                    'caption': f'Image {i}',
                    'bbox': [],
                    'category_id': 0
                })
            
            # Save synthetic annotations
            annotation_file = data_path / 'synthetic_annotations.json'
            with open(annotation_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            return MultiModalDataset(str(data_path), str(annotation_file), **kwargs)
    
    @staticmethod
    def combine_datasets(datasets: List[Dataset]) -> Dataset:
        """Combine multiple datasets into one."""
        if torch is None:
            raise ImportError("PyTorch required for dataset combination")
            
        return torch.utils.data.ConcatDataset(datasets)


# Example usage and testing
if __name__ == "__main__":
    # Test dataset creation
    print("Testing MultiModalDataset...")
    
    # Create dummy annotations for testing
    test_annotations = [
        {
            'image_id': 0,
            'image_path': 'test1.jpg',
            'caption': 'A test image with objects',
            'bbox': [10, 10, 100, 100],
            'category_id': 1,
            'text_regions': [
                {'text': 'STOP', 'bbox': [20, 20, 50, 30], 'confidence': 0.9}
            ],
            'questions': ['What is the main object?'],
            'answers': ['A sign']
        },
        {
            'image_id': 1,
            'image_path': 'test2.jpg', 
            'caption': 'Another test image',
            'bbox': [5, 5, 80, 80],
            'category_id': 2,
            'text_regions': [],
            'questions': ['What color is it?'],
            'answers': ['Blue']
        }
    ]
    
    # Save test annotations
    test_dir = Path('/tmp/test_dataset')
    test_dir.mkdir(exist_ok=True)
    
    with open(test_dir / 'annotations.json', 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    try:
        # Test dataset loading
        dataset = MultiModalDataset(
            str(test_dir),
            str(test_dir / 'annotations.json'),
            cache_images=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test different task datasets
        cap_dataset = CaptioningDataset(str(test_dir), str(test_dir / 'annotations.json'))
        ocr_dataset = OCRDataset(str(test_dir), str(test_dir / 'annotations.json'))
        vqa_dataset = VQADataset(str(test_dir), str(test_dir / 'annotations.json'))
        
        print(f"Captioning dataset: {len(cap_dataset)}")
        print(f"OCR dataset: {len(ocr_dataset)}")
        print(f"VQA dataset: {len(vqa_dataset)}")
        
        if torch is not None:
            # Test DataLoader
            dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)
            print(f"DataLoader created with {len(dataloader)} batches")
            
            # Test one batch
            for batch in dataloader:
                print(f"Batch keys: {batch.keys()}")
                if batch['images'] is not None:
                    print(f"Image batch shape: {batch['images'].shape}")
                break
        
        print("Dataset module test completed successfully!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()