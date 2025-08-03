"""Caching system for mobile multi-modal models."""

import hashlib
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class CacheManager:
    """Centralized cache management for mobile AI workflows."""
    
    def __init__(self, 
                 cache_dir: str = "cache/",
                 max_memory_mb: int = 512,
                 redis_url: Optional[str] = None,
                 enable_persistence: bool = True):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_memory_mb: Maximum memory usage for in-memory cache
            redis_url: Redis connection URL for distributed caching
            enable_persistence: Whether to enable disk-based persistence
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_persistence = enable_persistence
        
        # In-memory cache
        self._memory_cache = {}
        self._cache_sizes = {}
        self._access_times = {}
        self._lock = threading.RLock()
        
        # Redis cache (if available)
        self.redis_client = None
        if redis_url and redis is not None:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        
        # SQLite for metadata
        if enable_persistence:
            self.db_path = self.cache_dir / "cache_metadata.db"
            self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT,
                    size_bytes INTEGER,
                    created_at REAL,
                    accessed_at REAL,
                    hit_count INTEGER DEFAULT 0,
                    content_hash TEXT
                )
            """)
            conn.commit()
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for content addressing."""
        if isinstance(data, (str, bytes)):
            content = data if isinstance(data, bytes) else data.encode()
        elif isinstance(data, np.ndarray):
            content = data.tobytes()
        else:
            content = pickle.dumps(data)
        
        return hashlib.sha256(content).hexdigest()
    
    def _evict_lru(self):
        """Evict least recently used items from memory cache."""
        with self._lock:
            current_size = sum(self._cache_sizes.values())
            
            while current_size > self.max_memory_bytes and self._memory_cache:
                # Find LRU item
                lru_key = min(self._access_times.keys(), 
                             key=lambda k: self._access_times[k])
                
                # Remove from memory
                current_size -= self._cache_sizes.pop(lru_key, 0)
                self._memory_cache.pop(lru_key, None)
                self._access_times.pop(lru_key, None)
                
                logger.debug(f"Evicted {lru_key} from memory cache")
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
            
        Returns:
            True if successfully cached
        """
        try:
            current_time = time.time()
            content_hash = self._compute_hash(value)
            
            # Serialize value
            if isinstance(value, np.ndarray):
                serialized = value.tobytes()
                value_type = 'numpy'
                extra_info = {'dtype': str(value.dtype), 'shape': value.shape}
            else:
                serialized = pickle.dumps(value)
                value_type = 'pickle'
                extra_info = {}
            
            value_size = len(serialized)
            
            # Store in memory cache
            with self._lock:
                self._memory_cache[key] = {
                    'data': serialized,
                    'type': value_type,
                    'extra_info': extra_info,
                    'expires_at': current_time + ttl if ttl else None
                }
                self._cache_sizes[key] = value_size
                self._access_times[key] = current_time
                
                # Evict if necessary
                self._evict_lru()
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    redis_value = {
                        'data': serialized,
                        'type': value_type,
                        'extra_info': extra_info
                    }
                    self.redis_client.set(
                        key, 
                        pickle.dumps(redis_value),
                        ex=ttl
                    )
                except Exception as e:
                    logger.warning(f"Failed to store in Redis: {e}")
            
            # Store persistently
            if self.enable_persistence and value_size < 10 * 1024 * 1024:  # < 10MB
                file_path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
                
                try:
                    with open(file_path, 'wb') as f:
                        f.write(serialized)
                    
                    # Update database
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO cache_entries 
                            (key, file_path, size_bytes, created_at, accessed_at, content_hash)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (key, str(file_path), value_size, current_time, current_time, content_hash))
                        conn.commit()
                
                except Exception as e:
                    logger.warning(f"Failed to persist cache entry: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        current_time = time.time()
        
        # Check memory cache first
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check expiration
                if entry.get('expires_at') and current_time > entry['expires_at']:
                    self._memory_cache.pop(key, None)
                    self._cache_sizes.pop(key, None)
                    self._access_times.pop(key, None)
                else:
                    # Update access time
                    self._access_times[key] = current_time
                    
                    # Deserialize and return
                    return self._deserialize(entry)
        
        # Check Redis cache
        if self.redis_client:
            try:
                redis_data = self.redis_client.get(key)
                if redis_data:
                    entry = pickle.loads(redis_data)
                    
                    # Store back in memory cache
                    with self._lock:
                        self._memory_cache[key] = entry
                        self._cache_sizes[key] = len(entry['data'])
                        self._access_times[key] = current_time
                        self._evict_lru()
                    
                    return self._deserialize(entry)
                    
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Check persistent cache
        if self.enable_persistence:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT file_path, size_bytes FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        file_path, size_bytes = row
                        
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                data = f.read()
                            
                            # Try to determine type from database or guess
                            if data.startswith(b'\x80'):  # Pickle magic number
                                value_type = 'pickle'
                                extra_info = {}
                            else:
                                value_type = 'numpy'
                                extra_info = {}  # Would need to store this properly
                            
                            entry = {
                                'data': data,
                                'type': value_type,
                                'extra_info': extra_info
                            }
                            
                            # Update access time
                            conn.execute(
                                "UPDATE cache_entries SET accessed_at = ?, hit_count = hit_count + 1 WHERE key = ?",
                                (current_time, key)
                            )
                            conn.commit()
                            
                            # Store in memory for faster access
                            with self._lock:
                                self._memory_cache[key] = entry
                                self._cache_sizes[key] = size_bytes
                                self._access_times[key] = current_time
                                self._evict_lru()
                            
                            return self._deserialize(entry)
                        
            except Exception as e:
                logger.warning(f"Persistent cache error: {e}")
        
        return None
    
    def _deserialize(self, entry: Dict[str, Any]) -> Any:
        """Deserialize cached entry."""
        data = entry['data']
        value_type = entry['type']
        extra_info = entry.get('extra_info', {})
        
        if value_type == 'numpy':
            # Reconstruct numpy array
            dtype = np.dtype(extra_info.get('dtype', 'float32'))
            shape = extra_info.get('shape', (-1,))
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        else:
            # Unpickle
            return pickle.loads(data)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        found = False
        
        # Remove from memory
        with self._lock:
            if key in self._memory_cache:
                self._memory_cache.pop(key, None)
                self._cache_sizes.pop(key, None)
                self._access_times.pop(key, None)
                found = True
        
        # Remove from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        # Remove from persistent storage
        if self.enable_persistence:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT file_path FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        file_path = row[0]
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        found = True
                        
            except Exception as e:
                logger.warning(f"Persistent delete error: {e}")
        
        return found
    
    def clear(self):
        """Clear all cache entries."""
        # Clear memory
        with self._lock:
            self._memory_cache.clear()
            self._cache_sizes.clear()
            self._access_times.clear()
        
        # Clear Redis
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        
        # Clear persistent storage
        if self.enable_persistence:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Get all file paths and delete files
                    cursor = conn.execute("SELECT file_path FROM cache_entries")
                    for (file_path,) in cursor.fetchall():
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # Clear database
                    conn.execute("DELETE FROM cache_entries")
                    conn.commit()
                    
            except Exception as e:
                logger.warning(f"Persistent clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'memory_cache_size': len(self._memory_cache),
            'memory_usage_bytes': sum(self._cache_sizes.values()),
            'memory_usage_mb': sum(self._cache_sizes.values()) / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
        }
        
        # Redis stats
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats['redis_used_memory'] = redis_info.get('used_memory', 0)
                stats['redis_connected'] = True
            except Exception:
                stats['redis_connected'] = False
        else:
            stats['redis_connected'] = False
        
        # Persistent cache stats
        if self.enable_persistence:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as entry_count,
                            SUM(size_bytes) as total_size,
                            AVG(hit_count) as avg_hits
                        FROM cache_entries
                    """)
                    row = cursor.fetchone()
                    
                    if row:
                        stats['persistent_entries'] = row[0] or 0
                        stats['persistent_size_bytes'] = row[1] or 0
                        stats['persistent_size_mb'] = (row[1] or 0) / (1024 * 1024)
                        stats['avg_hit_count'] = row[2] or 0
                        
            except Exception as e:
                logger.warning(f"Error getting persistent stats: {e}")
        
        return stats


class ModelCache:
    """Specialized cache for model weights and components."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.model_prefix = "model:"
    
    def cache_model_weights(self, model_name: str, weights: Dict[str, Any]) -> bool:
        """Cache model weights."""
        key = f"{self.model_prefix}{model_name}:weights"
        return self.cache.put(key, weights)
    
    def get_model_weights(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached model weights."""
        key = f"{self.model_prefix}{model_name}:weights"
        return self.cache.get(key)
    
    def cache_model_config(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Cache model configuration."""
        key = f"{self.model_prefix}{model_name}:config"
        return self.cache.put(key, config)
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached model configuration."""
        key = f"{self.model_prefix}{model_name}:config"
        return self.cache.get(key)
    
    def cache_quantized_model(self, model_name: str, quantized_data: bytes) -> bool:
        """Cache quantized model."""
        key = f"{self.model_prefix}{model_name}:quantized"
        return self.cache.put(key, quantized_data)
    
    def get_quantized_model(self, model_name: str) -> Optional[bytes]:
        """Retrieve cached quantized model."""
        key = f"{self.model_prefix}{model_name}:quantized"
        return self.cache.get(key)


class FeatureCache:
    """Cache for computed features and embeddings."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.feature_prefix = "features:"
    
    def cache_image_features(self, image_hash: str, features: np.ndarray) -> bool:
        """Cache image features."""
        key = f"{self.feature_prefix}image:{image_hash}"
        return self.cache.put(key, features, ttl=3600)  # 1 hour TTL
    
    def get_image_features(self, image_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached image features."""
        key = f"{self.feature_prefix}image:{image_hash}"
        return self.cache.get(key)
    
    def cache_text_embeddings(self, text_hash: str, embeddings: np.ndarray) -> bool:
        """Cache text embeddings."""
        key = f"{self.feature_prefix}text:{text_hash}"
        return self.cache.put(key, embeddings, ttl=3600)
    
    def get_text_embeddings(self, text_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached text embeddings."""
        key = f"{self.feature_prefix}text:{text_hash}"
        return self.cache.get(key)
    
    def cache_inference_result(self, input_hash: str, result: Dict[str, Any]) -> bool:
        """Cache inference result."""
        key = f"{self.feature_prefix}inference:{input_hash}"
        return self.cache.put(key, result, ttl=1800)  # 30 minutes TTL
    
    def get_inference_result(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached inference result."""
        key = f"{self.feature_prefix}inference:{input_hash}"
        return self.cache.get(key)


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # Test cache manager
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Testing CacheManager...")
        
        # Create cache manager
        cache = CacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            max_memory_mb=50,
            enable_persistence=True
        )
        
        # Test basic operations
        test_data = {"message": "Hello, cache!", "number": 42}
        
        # Store and retrieve
        assert cache.put("test_key", test_data) == True
        retrieved = cache.get("test_key")
        assert retrieved == test_data
        print("✓ Basic cache operations work")
        
        # Test numpy arrays
        test_array = np.random.rand(100, 100).astype(np.float32)
        cache.put("test_array", test_array)
        retrieved_array = cache.get("test_array")
        assert np.allclose(test_array, retrieved_array)
        print("✓ Numpy array caching works")
        
        # Test TTL
        cache.put("ttl_test", "temporary", ttl=1)
        assert cache.get("ttl_test") == "temporary"
        time.sleep(1.1)
        assert cache.get("ttl_test") is None
        print("✓ TTL expiration works")
        
        # Test model cache
        model_cache = ModelCache(cache)
        weights = {"layer1": np.random.rand(10, 10), "layer2": np.random.rand(5, 5)}
        model_cache.cache_model_weights("test_model", weights)
        retrieved_weights = model_cache.get_model_weights("test_model")
        assert len(retrieved_weights) == 2
        print("✓ Model cache works")
        
        # Test feature cache
        feature_cache = FeatureCache(cache)
        features = np.random.rand(512)
        feature_cache.cache_image_features("image_hash_123", features)
        retrieved_features = feature_cache.get_image_features("image_hash_123")
        assert np.allclose(features, retrieved_features)
        print("✓ Feature cache works")
        
        # Test statistics
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        
        # Test clearing
        cache.clear()
        assert cache.get("test_key") is None
        print("✓ Cache clearing works")
        
        print("All cache tests passed!")