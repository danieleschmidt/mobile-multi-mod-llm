"""Storage systems for mobile multi-modal models and data."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

try:
    from google.cloud import storage as gcs
except ImportError:
    gcs = None

logger = logging.getLogger(__name__)


class DataStorage:
    """Base storage interface for datasets and processed data."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize data storage.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Standard directory structure
        self.raw_data_dir = self.base_path / "raw"
        self.processed_data_dir = self.base_path / "processed"
        self.cache_dir = self.base_path / "cache"
        self.temp_dir = self.base_path / "temp"
        
        for directory in [self.raw_data_dir, self.processed_data_dir, self.cache_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"Initialized data storage at {self.base_path}")
    
    def store_raw_data(self, dataset_name: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """Store raw dataset."""
        dataset_dir = self.raw_data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Store data
        if isinstance(data, dict):
            data_file = dataset_dir / "data.json"
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, (list, np.ndarray)):
            data_file = dataset_dir / "data.npy"
            np.save(data_file, data)
        else:
            data_file = dataset_dir / "data.bin"
            with open(data_file, 'wb') as f:
                if hasattr(data, 'tobytes'):
                    f.write(data.tobytes())
                else:
                    f.write(str(data).encode())
        
        # Store metadata
        if metadata:
            metadata_file = dataset_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Stored raw data for {dataset_name} at {dataset_dir}")
        return str(data_file)
    
    def load_raw_data(self, dataset_name: str) -> Optional[Any]:
        """Load raw dataset."""
        dataset_dir = self.raw_data_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset {dataset_name} not found")
            return None
        
        # Try different file formats
        for file_name, loader in [
            ("data.json", self._load_json),
            ("data.npy", self._load_numpy),
            ("data.bin", self._load_binary)
        ]:
            file_path = dataset_dir / file_name
            if file_path.exists():
                return loader(file_path)
        
        logger.warning(f"No data files found for {dataset_name}")
        return None
    
    def store_processed_data(self, dataset_name: str, split: str, data: Any, 
                           processing_info: Optional[Dict] = None) -> str:
        """Store processed dataset split."""
        split_dir = self.processed_data_dir / dataset_name / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Store data based on type
        if isinstance(data, dict):
            data_file = split_dir / "data.json"
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, (list, np.ndarray)):
            data_file = split_dir / "data.npy"
            np.save(data_file, data)
        else:
            data_file = split_dir / "data.pkl"
            import pickle
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
        
        # Store processing info
        if processing_info:
            info_file = split_dir / "processing_info.json"
            with open(info_file, 'w') as f:
                json.dump(processing_info, f, indent=2)
        
        logger.info(f"Stored processed data for {dataset_name}/{split}")
        return str(data_file)
    
    def load_processed_data(self, dataset_name: str, split: str) -> Optional[Any]:
        """Load processed dataset split."""
        split_dir = self.processed_data_dir / dataset_name / split
        
        if not split_dir.exists():
            logger.warning(f"Processed data {dataset_name}/{split} not found")
            return None
        
        # Try different file formats
        for file_name, loader in [
            ("data.json", self._load_json),
            ("data.npy", self._load_numpy),
            ("data.pkl", self._load_pickle)
        ]:
            file_path = split_dir / file_name
            if file_path.exists():
                return loader(file_path)
        
        return None
    
    def list_datasets(self, include_processed: bool = True) -> Dict[str, List[str]]:
        """List available datasets."""
        datasets = {"raw": [], "processed": {}}
        
        # List raw datasets
        if self.raw_data_dir.exists():
            datasets["raw"] = [d.name for d in self.raw_data_dir.iterdir() if d.is_dir()]
        
        # List processed datasets
        if include_processed and self.processed_data_dir.exists():
            for dataset_dir in self.processed_data_dir.iterdir():
                if dataset_dir.is_dir():
                    splits = [s.name for s in dataset_dir.iterdir() if s.is_dir()]
                    datasets["processed"][dataset_dir.name] = splits
        
        return datasets
    
    def delete_dataset(self, dataset_name: str, split: Optional[str] = None):
        """Delete dataset or split."""
        if split:
            # Delete specific split
            split_dir = self.processed_data_dir / dataset_name / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
                logger.info(f"Deleted {dataset_name}/{split}")
        else:
            # Delete entire dataset
            raw_dir = self.raw_data_dir / dataset_name
            processed_dir = self.processed_data_dir / dataset_name
            
            for directory in [raw_dir, processed_dir]:
                if directory.exists():
                    shutil.rmtree(directory)
            
            logger.info(f"Deleted dataset {dataset_name}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        def get_dir_size(path: Path) -> int:
            total_size = 0
            if path.exists():
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            return total_size
        
        stats = {
            "raw_data_bytes": get_dir_size(self.raw_data_dir),
            "processed_data_bytes": get_dir_size(self.processed_data_dir),
            "cache_bytes": get_dir_size(self.cache_dir),
            "temp_bytes": get_dir_size(self.temp_dir),
        }
        
        # Convert to MB
        for key in stats:
            stats[key.replace('_bytes', '_mb')] = stats[key] / (1024 * 1024)
        
        stats["total_bytes"] = sum(v for k, v in stats.items() if k.endswith('_bytes'))
        stats["total_mb"] = stats["total_bytes"] / (1024 * 1024)
        
        return stats
    
    def cleanup_temp(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir()
        logger.info("Cleaned up temporary files")
    
    def _load_json(self, file_path: Path) -> Any:
        """Load JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _load_numpy(self, file_path: Path) -> np.ndarray:
        """Load numpy file."""
        return np.load(file_path)
    
    def _load_binary(self, file_path: Path) -> bytes:
        """Load binary file."""
        with open(file_path, 'rb') as f:
            return f.read()
    
    def _load_pickle(self, file_path: Path) -> Any:
        """Load pickle file."""
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class ModelStorage:
    """Storage system for model artifacts and checkpoints."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize model storage."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Model storage structure
        self.models_dir = self.base_path / "models"
        self.checkpoints_dir = self.base_path / "checkpoints"
        self.exports_dir = self.base_path / "exports"
        self.configs_dir = self.base_path / "configs"
        
        for directory in [self.models_dir, self.checkpoints_dir, self.exports_dir, self.configs_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"Initialized model storage at {self.base_path}")
    
    def save_model(self, model_name: str, model_data: Dict[str, Any], 
                  model_config: Optional[Dict] = None) -> str:
        """Save trained model."""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model data (weights, etc.)
        if 'torch' in str(type(model_data)):
            # PyTorch model
            import torch
            model_file = model_dir / f"{model_name}.pth"
            torch.save(model_data, model_file)
        else:
            # Generic model data
            import pickle
            model_file = model_dir / f"{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
        
        # Save configuration
        if model_config:
            config_file = model_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(model_config, f, indent=2)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "saved_at": str(pd.Timestamp.now()) if 'pd' in globals() else str(time.time()),
            "file_size_bytes": model_file.stat().st_size,
            "model_type": type(model_data).__name__
        }
        
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model {model_name} to {model_file}")
        return str(model_file)
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load trained model."""
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            logger.warning(f"Model {model_name} not found")
            return None
        
        # Try different file formats
        for file_ext, loader in [
            (".pth", self._load_torch),
            (".pkl", self._load_pickle),
            (".onnx", self._load_onnx)
        ]:
            model_file = model_dir / f"{model_name}{file_ext}"
            if model_file.exists():
                return loader(model_file)
        
        logger.warning(f"No model files found for {model_name}")
        return None
    
    def save_checkpoint(self, model_name: str, epoch: int, checkpoint_data: Dict[str, Any]) -> str:
        """Save training checkpoint."""
        checkpoint_dir = self.checkpoints_dir / model_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        
        # Add metadata to checkpoint
        checkpoint_data.update({
            "epoch": epoch,
            "model_name": model_name,
            "saved_at": str(time.time()) if 'time' in globals() else "unknown"
        })
        
        # Save checkpoint
        if 'torch' in str(type(checkpoint_data.get('model_state_dict', {}))):
            import torch
            torch.save(checkpoint_data, checkpoint_file)
        else:
            import pickle
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint for {model_name} epoch {epoch}")
        return str(checkpoint_file)
    
    def load_checkpoint(self, model_name: str, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load training checkpoint."""
        checkpoint_dir = self.checkpoints_dir / model_name
        
        if not checkpoint_dir.exists():
            logger.warning(f"No checkpoints found for {model_name}")
            return None
        
        if epoch is not None:
            # Load specific epoch
            checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            if checkpoint_file.exists():
                return self._load_torch(checkpoint_file) if checkpoint_file.suffix == '.pth' else self._load_pickle(checkpoint_file)
        else:
            # Load latest checkpoint
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
            if checkpoint_files:
                latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                return self._load_torch(latest_file) if latest_file.suffix == '.pth' else self._load_pickle(latest_file)
        
        return None
    
    def export_model(self, model_name: str, model, export_format: str = "onnx", 
                    export_config: Optional[Dict] = None) -> str:
        """Export model to mobile format."""
        export_dir = self.exports_dir / model_name
        export_dir.mkdir(exist_ok=True)
        
        export_file = export_dir / f"{model_name}.{export_format}"
        
        if export_format.lower() == "onnx":
            self._export_onnx(model, export_file, export_config)
        elif export_format.lower() == "tflite":
            self._export_tflite(model, export_file, export_config)
        elif export_format.lower() == "coreml":
            self._export_coreml(model, export_file, export_config)
        elif export_format.lower() == "torchscript":
            self._export_torchscript(model, export_file, export_config)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Exported {model_name} to {export_format} format")
        return str(export_file)
    
    def list_models(self) -> Dict[str, List[str]]:
        """List available models and checkpoints."""
        models = {"trained": [], "checkpoints": {}, "exports": {}}
        
        # List trained models
        if self.models_dir.exists():
            models["trained"] = [d.name for d in self.models_dir.iterdir() if d.is_dir()]
        
        # List checkpoints
        if self.checkpoints_dir.exists():
            for model_dir in self.checkpoints_dir.iterdir():
                if model_dir.is_dir():
                    checkpoints = [f.stem for f in model_dir.glob("checkpoint_epoch_*.pth")]
                    models["checkpoints"][model_dir.name] = checkpoints
        
        # List exports
        if self.exports_dir.exists():
            for model_dir in self.exports_dir.iterdir():
                if model_dir.is_dir():
                    exports = [f.name for f in model_dir.iterdir() if f.is_file()]
                    models["exports"][model_dir.name] = exports
        
        return models
    
    def _load_torch(self, file_path: Path) -> Any:
        """Load PyTorch model/checkpoint."""
        try:
            import torch
            return torch.load(file_path, map_location='cpu')
        except ImportError:
            logger.error("PyTorch not available for loading .pth files")
            return None
    
    def _load_pickle(self, file_path: Path) -> Any:
        """Load pickle file."""
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_onnx(self, file_path: Path) -> Any:
        """Load ONNX model."""
        try:
            import onnx
            return onnx.load(str(file_path))
        except ImportError:
            logger.error("ONNX not available")
            return None
    
    def _export_onnx(self, model, export_file: Path, config: Optional[Dict]):
        """Export to ONNX format."""
        try:
            import torch
            
            # Get export configuration
            input_shape = config.get('input_shape', (1, 3, 224, 224)) if config else (1, 3, 224, 224)
            opset_version = config.get('opset_version', 16) if config else 16
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Export
            torch.onnx.export(
                model,
                dummy_input,
                str(export_file),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _export_tflite(self, model, export_file: Path, config: Optional[Dict]):
        """Export to TensorFlow Lite format."""
        logger.warning("TensorFlow Lite export not implemented")
        raise NotImplementedError("TensorFlow Lite export not implemented")
    
    def _export_coreml(self, model, export_file: Path, config: Optional[Dict]):
        """Export to Core ML format."""
        logger.warning("Core ML export not implemented")
        raise NotImplementedError("Core ML export not implemented")
    
    def _export_torchscript(self, model, export_file: Path, config: Optional[Dict]):
        """Export to TorchScript format."""
        try:
            import torch
            
            # Trace or script the model
            if config and config.get('use_script', False):
                scripted_model = torch.jit.script(model)
            else:
                input_shape = config.get('input_shape', (1, 3, 224, 224)) if config else (1, 3, 224, 224)
                dummy_input = torch.randn(*input_shape)
                scripted_model = torch.jit.trace(model, dummy_input)
            
            # Optimize for mobile if specified
            if config and config.get('optimize_for_mobile', True):
                scripted_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
            
            # Save
            scripted_model.save(str(export_file))
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise


class CloudStorage:
    """Cloud storage interface for distributed model storage."""
    
    def __init__(self, provider: str, bucket_name: str, **kwargs):
        """Initialize cloud storage.
        
        Args:
            provider: Cloud provider ('aws', 'gcp', 'azure')
            bucket_name: Storage bucket name
            **kwargs: Provider-specific configuration
        """
        self.provider = provider.lower()
        self.bucket_name = bucket_name
        self.client = None
        
        if self.provider == 'aws':
            self._init_aws(**kwargs)
        elif self.provider == 'gcp':
            self._init_gcp(**kwargs)
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
    
    def _init_aws(self, **kwargs):
        """Initialize AWS S3 client."""
        if boto3 is None:
            raise ImportError("boto3 required for AWS storage")
        
        self.client = boto3.client(
            's3',
            aws_access_key_id=kwargs.get('access_key_id'),
            aws_secret_access_key=kwargs.get('secret_access_key'),
            region_name=kwargs.get('region', 'us-east-1')
        )
        
        # Create bucket if it doesn't exist
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket_name)
        
        logger.info(f"Initialized AWS S3 storage: {self.bucket_name}")
    
    def _init_gcp(self, **kwargs):
        """Initialize Google Cloud Storage client."""
        if gcs is None:
            raise ImportError("google-cloud-storage required for GCP storage")
        
        self.client = gcs.Client(project=kwargs.get('project_id'))
        self.bucket = self.client.bucket(self.bucket_name)
        
        logger.info(f"Initialized GCP storage: {self.bucket_name}")
    
    def upload_model(self, local_path: str, remote_key: str) -> bool:
        """Upload model to cloud storage."""
        try:
            if self.provider == 'aws':
                self.client.upload_file(local_path, self.bucket_name, remote_key)
            elif self.provider == 'gcp':
                blob = self.bucket.blob(remote_key)
                blob.upload_from_filename(local_path)
            
            logger.info(f"Uploaded {local_path} to {remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def download_model(self, remote_key: str, local_path: str) -> bool:
        """Download model from cloud storage."""
        try:
            if self.provider == 'aws':
                self.client.download_file(self.bucket_name, remote_key, local_path)
            elif self.provider == 'gcp':
                blob = self.bucket.blob(remote_key)
                blob.download_to_filename(local_path)
            
            logger.info(f"Downloaded {remote_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def list_models(self, prefix: str = "") -> List[str]:
        """List models in cloud storage."""
        try:
            if self.provider == 'aws':
                response = self.client.list_objects_v2(
                    Bucket=self.bucket_name, 
                    Prefix=prefix
                )
                return [obj['Key'] for obj in response.get('Contents', [])]
            elif self.provider == 'gcp':
                blobs = self.client.list_blobs(self.bucket, prefix=prefix)
                return [blob.name for blob in blobs]
            
        except Exception as e:
            logger.error(f"List failed: {e}")
            return []
    
    def delete_model(self, remote_key: str) -> bool:
        """Delete model from cloud storage."""
        try:
            if self.provider == 'aws':
                self.client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            elif self.provider == 'gcp':
                blob = self.bucket.blob(remote_key)
                blob.delete()
            
            logger.info(f"Deleted {remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import time
    
    # Test data storage
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Testing DataStorage...")
        
        storage = DataStorage(temp_dir)
        
        # Test raw data storage
        test_data = {"samples": [1, 2, 3, 4, 5], "labels": ["a", "b", "c", "d", "e"]}
        storage.store_raw_data("test_dataset", test_data, {"version": "1.0"})
        
        loaded_data = storage.load_raw_data("test_dataset")
        assert loaded_data == test_data
        print("✓ Raw data storage works")
        
        # Test processed data storage
        processed_data = np.random.rand(100, 10)
        storage.store_processed_data("test_dataset", "train", processed_data)
        
        loaded_processed = storage.load_processed_data("test_dataset", "train")
        assert np.allclose(processed_data, loaded_processed)
        print("✓ Processed data storage works")
        
        # Test listing
        datasets = storage.list_datasets()
        assert "test_dataset" in datasets["raw"]
        assert "test_dataset" in datasets["processed"]
        print("✓ Dataset listing works")
        
        # Test stats
        stats = storage.get_storage_stats()
        assert stats["total_bytes"] > 0
        print(f"Storage stats: {stats}")
        
        print("✓ Data storage tests passed")
    
    # Test model storage
    with tempfile.TemporaryDirectory() as temp_dir:
        print("\nTesting ModelStorage...")
        
        model_storage = ModelStorage(temp_dir)
        
        # Test model saving/loading
        test_model = {"weights": np.random.rand(10, 10), "bias": np.random.rand(10)}
        model_storage.save_model("test_model", test_model, {"architecture": "simple"})
        
        loaded_model = model_storage.load_model("test_model")
        assert np.allclose(test_model["weights"], loaded_model["weights"])
        print("✓ Model storage works")
        
        # Test checkpoint saving/loading
        checkpoint = {
            "model_state_dict": test_model,
            "optimizer_state_dict": {"lr": 0.001},
            "loss": 0.5
        }
        model_storage.save_checkpoint("test_model", 10, checkpoint)
        
        loaded_checkpoint = model_storage.load_checkpoint("test_model", 10)
        assert loaded_checkpoint["epoch"] == 10
        print("✓ Checkpoint storage works")
        
        # Test listing
        models = model_storage.list_models()
        assert "test_model" in models["trained"]
        print("✓ Model listing works")
        
        print("✓ Model storage tests passed")
    
    print("\nAll storage tests completed successfully!")