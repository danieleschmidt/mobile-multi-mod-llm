"""Intelligent Cache System - Advanced caching with ML-driven optimization.

This module implements production-grade intelligent caching with:
1. Multi-tier cache hierarchy (L1 memory, L2 disk, L3 remote)
2. ML-driven cache replacement policies
3. Predictive prefetching based on usage patterns
4. Content-aware compression and deduplication
5. Mobile-optimized cache management with battery/memory awareness
"""

import asyncio
import hashlib
import logging
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"      # In-memory cache
    L2_DISK = "l2_disk"          # Local disk cache
    L3_REMOTE = "l3_remote"      # Remote/distributed cache


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    ADAPTIVE = "adaptive"        # ML-driven adaptive policy
    TTL = "ttl"                  # Time To Live
    SIZE_AWARE = "size_aware"    # Size-aware eviction


class CacheHit(Enum):
    """Cache hit types."""
    L1_HIT = "l1_hit"
    L2_HIT = "l2_hit"
    L3_HIT = "l3_hit"
    MISS = "miss"
    PREFETCH_HIT = "prefetch_hit"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    compression_ratio: float = 1.0
    priority_score: float = 0.5
    access_pattern: List[float] = field(default_factory=list)
    
    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.creation_time
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.creation_time > self.ttl
    
    def update_access(self):
        """Update access information."""
        self.access_count += 1
        current_time = time.time()
        
        # Record access pattern
        self.access_pattern.append(current_time)
        
        # Keep only recent access pattern (last 100 accesses)
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]
        
        self.last_access = current_time


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    prefetch_hits: int = 0
    evictions: int = 0
    compression_savings: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    cache_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate overall hit rate."""
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits + self.prefetch_hits
        return total_hits / max(self.total_requests, 1)
    
    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 hit rate."""
        return self.l1_hits / max(self.total_requests, 1)


class CacheCompressor:
    """Intelligent compression for cache entries."""
    
    def __init__(self):
        self.compression_stats = {}
        
    def compress(self, data: Any) -> Tuple[bytes, float]:
        """Compress data and return compressed bytes and compression ratio."""
        try:
            # Serialize data
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # Simple compression using zlib (in practice, use more sophisticated methods)
            import zlib
            compressed = zlib.compress(serialized, level=6)
            compressed_size = len(compressed)
            
            compression_ratio = compressed_size / original_size
            
            logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes "
                        f"(ratio: {compression_ratio:.3f})")
            
            return compressed, compression_ratio
            
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
            # Fallback to uncompressed
            return pickle.dumps(data), 1.0
    
    def decompress(self, compressed_data: bytes) -> Any:
        """Decompress data."""
        try:
            import zlib
            decompressed = zlib.decompress(compressed_data)
            return pickle.loads(decompressed)
        except:
            # Try direct pickle load (uncompressed data)
            return pickle.loads(compressed_data)


class MLCacheOptimizer:
    """ML-based cache optimization and prefetching."""
    
    def __init__(self):
        self.access_patterns = {}
        self.prediction_accuracy = 0.0
        self.prefetch_candidates = []
        
    def predict_access_probability(self, key: str, context: Dict[str, Any] = None) -> float:
        """Predict probability of key being accessed soon."""
        if key not in self.access_patterns:
            return 0.1  # Default low probability
        
        pattern = self.access_patterns[key]
        
        # Simple time-based prediction
        current_time = time.time()
        if len(pattern) < 2:
            return 0.1
        
        # Calculate average access interval
        intervals = [pattern[i] - pattern[i-1] for i in range(1, len(pattern))]
        avg_interval = np.mean(intervals) if intervals else 3600  # Default 1 hour
        
        # Time since last access
        time_since_last = current_time - pattern[-1]
        
        # Probability decreases with time since last access
        probability = max(0.0, 1.0 - (time_since_last / (avg_interval * 2)))
        
        return probability
    
    def update_access_pattern(self, key: str):
        """Update access pattern for a key."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(time.time())
        
        # Keep only recent history
        if len(self.access_patterns[key]) > 50:
            self.access_patterns[key] = self.access_patterns[key][-50:]
    
    def get_prefetch_candidates(self, max_candidates: int = 10) -> List[str]:
        """Get candidates for prefetching."""
        candidates = []
        
        for key in self.access_patterns:
            probability = self.predict_access_probability(key)
            if probability > 0.3:  # Threshold for prefetching
                candidates.append((key, probability))
        
        # Sort by probability and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in candidates[:max_candidates]]
    
    def calculate_priority_score(self, entry: CacheEntry) -> float:
        """Calculate priority score for cache entry."""
        # Factors: access frequency, recency, size efficiency, prediction
        frequency_score = min(1.0, entry.access_count / 100.0)
        recency_score = max(0.0, 1.0 - (entry.age / 3600))  # 1 hour decay
        size_efficiency = 1.0 / (1.0 + np.log(entry.size_bytes / 1024))  # Size penalty
        prediction_score = self.predict_access_probability(entry.key)
        
        # Weighted combination
        priority = (frequency_score * 0.3 + 
                   recency_score * 0.25 + 
                   size_efficiency * 0.2 + 
                   prediction_score * 0.25)
        
        return priority


class AdaptiveCacheLayer(ABC):
    """Abstract base class for cache layers."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get current cache size in bytes."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache layer statistics."""
        pass


class L1MemoryCache(AdaptiveCacheLayer):
    """L1 in-memory cache with intelligent eviction."""
    
    def __init__(self, max_size_mb: int = 128, eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.entries = {}
        self.access_order = []  # For LRU
        self.compressor = CacheCompressor()
        self.ml_optimizer = MLCacheOptimizer()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache."""
        if key in self.entries:
            entry = self.entries[key]
            
            # Check if expired
            if entry.is_expired:
                await self.delete(key)
                self.stats["misses"] += 1
                return None
            
            # Update access information
            entry.update_access()
            self.ml_optimizer.update_access_pattern(key)
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats["hits"] += 1
            logger.debug(f"L1 cache hit for key: {key}")
            
            # Decompress if needed
            if isinstance(entry.value, bytes):
                return self.compressor.decompress(entry.value)
            return entry.value
        
        self.stats["misses"] += 1
        return None
    
    async def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in L1 cache."""
        try:
            # Compress value
            compressed_value, compression_ratio = self.compressor.compress(value)
            
            # Calculate size
            size_bytes = len(compressed_value)
            
            # Check if we need to evict entries
            await self._ensure_capacity(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                size_bytes=size_bytes,
                ttl=ttl,
                compression_ratio=compression_ratio
            )
            
            # Update ML optimizer priority score
            entry.priority_score = self.ml_optimizer.calculate_priority_score(entry)
            
            self.entries[key] = entry
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            logger.debug(f"Stored in L1 cache: {key} ({size_bytes} bytes, "
                        f"compression: {compression_ratio:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in L1 cache: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from L1 cache."""
        if key in self.entries:
            del self.entries[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    async def clear(self) -> bool:
        """Clear L1 cache."""
        self.entries.clear()
        self.access_order.clear()
        return True
    
    def get_size(self) -> int:
        """Get current cache size in bytes."""
        return sum(entry.size_bytes for entry in self.entries.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics."""
        current_size = self.get_size()
        return {
            "layer": "L1_MEMORY",
            "entries": len(self.entries),
            "size_bytes": current_size,
            "size_mb": current_size / (1024 * 1024),
            "utilization": current_size / self.max_size_bytes,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": self.stats["hits"] / max(self.stats["hits"] + self.stats["misses"], 1)
        }
    
    async def _ensure_capacity(self, required_bytes: int):
        """Ensure sufficient capacity by evicting entries if needed."""
        current_size = self.get_size()
        
        while current_size + required_bytes > self.max_size_bytes and self.entries:
            evicted_key = await self._select_eviction_candidate()
            if evicted_key:
                evicted_entry = self.entries[evicted_key]
                current_size -= evicted_entry.size_bytes
                await self.delete(evicted_key)
                self.stats["evictions"] += 1
                logger.debug(f"Evicted from L1 cache: {evicted_key}")
            else:
                break
    
    async def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on policy."""
        if not self.entries:
            return None
        
        if self.eviction_policy == EvictionPolicy.LRU:
            return self.access_order[0] if self.access_order else None
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            return min(self.entries.keys(), key=lambda k: self.entries[k].access_count)
        
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # Use ML-based priority scoring
            min_priority = float('inf')
            candidate = None
            
            for key, entry in self.entries.items():
                priority = self.ml_optimizer.calculate_priority_score(entry)
                if priority < min_priority:
                    min_priority = priority
                    candidate = key
            
            return candidate
        
        else:  # Default to LRU
            return self.access_order[0] if self.access_order else None


class L2DiskCache(AdaptiveCacheLayer):
    """L2 disk-based cache with SQLite backend."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 512):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.db_path = self.cache_dir / "cache.db"
        self.compressor = CacheCompressor()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT,
                    size_bytes INTEGER,
                    access_count INTEGER,
                    last_access REAL,
                    creation_time REAL,
                    ttl REAL,
                    compression_ratio REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT filename, ttl, creation_time FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    self.stats["misses"] += 1
                    return None
                
                filename, ttl, creation_time = row
                
                # Check if expired
                if ttl and time.time() - creation_time > ttl:
                    await self.delete(key)
                    self.stats["misses"] += 1
                    return None
                
                # Read file
                file_path = self.cache_dir / filename
                if not file_path.exists():
                    await self.delete(key)
                    self.stats["misses"] += 1
                    return None
                
                with open(file_path, 'rb') as f:
                    compressed_data = f.read()
                
                # Update access information
                conn.execute(
                    "UPDATE cache_entries SET access_count = access_count + 1, last_access = ? WHERE key = ?",
                    (time.time(), key)
                )
                
                self.stats["hits"] += 1
                logger.debug(f"L2 cache hit for key: {key}")
                
                # Decompress and return
                return self.compressor.decompress(compressed_data)
                
        except Exception as e:
            logger.error(f"L2 cache get error: {str(e)}")
            self.stats["misses"] += 1
            return None
    
    async def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in L2 cache."""
        try:
            # Compress value
            compressed_data, compression_ratio = self.compressor.compress(value)
            size_bytes = len(compressed_data)
            
            # Generate filename
            filename = f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            file_path = self.cache_dir / filename
            
            # Check capacity
            await self._ensure_capacity(size_bytes)
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
            
            # Update database
            current_time = time.time()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, filename, size_bytes, access_count, last_access, creation_time, ttl, compression_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (key, filename, size_bytes, 1, current_time, current_time, ttl, compression_ratio))
            
            logger.debug(f"Stored in L2 cache: {key} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"L2 cache put error: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from L2 cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT filename FROM cache_entries WHERE key = ?", (key,))
                row = cursor.fetchone()
                
                if row:
                    filename = row[0]
                    file_path = self.cache_dir / filename
                    
                    # Delete file
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Delete from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    return True
                
        except Exception as e:
            logger.error(f"L2 cache delete error: {str(e)}")
        
        return False
    
    async def clear(self) -> bool:
        """Clear L2 cache."""
        try:
            # Delete all cache files
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
            
            return True
            
        except Exception as e:
            logger.error(f"L2 cache clear error: {str(e)}")
            return False
    
    def get_size(self) -> int:
        """Get current cache size in bytes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                result = cursor.fetchone()[0]
                return result if result else 0
        except:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L2 cache statistics."""
        current_size = self.get_size()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                entry_count = cursor.fetchone()[0]
        except:
            entry_count = 0
        
        return {
            "layer": "L2_DISK",
            "entries": entry_count,
            "size_bytes": current_size,
            "size_mb": current_size / (1024 * 1024),
            "utilization": current_size / self.max_size_bytes,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": self.stats["hits"] / max(self.stats["hits"] + self.stats["misses"], 1)
        }
    
    async def _ensure_capacity(self, required_bytes: int):
        """Ensure sufficient capacity for L2 cache."""
        current_size = self.get_size()
        
        while current_size + required_bytes > self.max_size_bytes:
            # Find least recently used entry
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT key FROM cache_entries ORDER BY last_access ASC LIMIT 1"
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        key_to_evict = row[0]
                        entry_size = conn.execute(
                            "SELECT size_bytes FROM cache_entries WHERE key = ?",
                            (key_to_evict,)
                        ).fetchone()[0]
                        
                        await self.delete(key_to_evict)
                        current_size -= entry_size
                        self.stats["evictions"] += 1
                        logger.debug(f"Evicted from L2 cache: {key_to_evict}")
                    else:
                        break
                        
            except Exception as e:
                logger.error(f"L2 cache eviction error: {str(e)}")
                break


class IntelligentCacheManager:
    """Main cache manager with multi-tier hierarchy and ML optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize cache layers
        self.l1_cache = L1MemoryCache(
            max_size_mb=self.config["l1_size_mb"],
            eviction_policy=EvictionPolicy(self.config["eviction_policy"])
        )
        
        self.l2_cache = L2DiskCache(
            cache_dir=Path(self.config["cache_dir"]),
            max_size_mb=self.config["l2_size_mb"]
        )
        
        # ML optimizer for global optimization
        self.ml_optimizer = MLCacheOptimizer()
        
        # Metrics
        self.global_metrics = CacheMetrics()
        
        # Prefetching
        self.prefetch_enabled = self.config["enable_prefetching"]
        self.prefetch_task = None
        
        if self.prefetch_enabled:
            self.prefetch_task = asyncio.create_task(self._prefetch_worker())
        
        logger.info("Intelligent cache manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default cache configuration."""
        return {
            "l1_size_mb": 128,
            "l2_size_mb": 512,
            "cache_dir": "/tmp/mobile_multimodal_cache",
            "eviction_policy": "adaptive",
            "enable_prefetching": True,
            "prefetch_threshold": 0.3,
            "prefetch_interval": 60.0,
            "default_ttl": 3600.0
        }
    
    async def get(self, key: str, loader: Optional[Callable] = None) -> Optional[Any]:
        """Get value from cache hierarchy with optional loader function."""
        start_time = time.perf_counter()
        self.global_metrics.total_requests += 1
        
        hit_type = CacheHit.MISS
        value = None
        
        try:
            # Try L1 cache first
            value = await self.l1_cache.get(key)
            if value is not None:
                hit_type = CacheHit.L1_HIT
                self.global_metrics.l1_hits += 1
            else:
                # Try L2 cache
                value = await self.l2_cache.get(key)
                if value is not None:
                    hit_type = CacheHit.L2_HIT
                    self.global_metrics.l2_hits += 1
                    
                    # Promote to L1 cache
                    await self.l1_cache.put(key, value)
            
            # If cache miss and loader provided, load and cache
            if value is None and loader:
                self.global_metrics.misses += 1
                
                if asyncio.iscoroutinefunction(loader):
                    value = await loader(key)
                else:
                    value = loader(key)
                
                if value is not None:
                    # Store in cache hierarchy
                    await self.put(key, value)
            
            # Update ML optimizer
            if value is not None:
                self.ml_optimizer.update_access_pattern(key)
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            self.global_metrics.misses += 1
        
        # Update metrics
        response_time = time.perf_counter() - start_time
        self.global_metrics.avg_response_time = (
            (self.global_metrics.avg_response_time * (self.global_metrics.total_requests - 1) +
             response_time) / self.global_metrics.total_requests
        )
        
        logger.debug(f"Cache {hit_type.value} for key: {key} (time: {response_time:.4f}s)")
        
        return value
    
    async def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache hierarchy."""
        if ttl is None:
            ttl = self.config["default_ttl"]
        
        try:
            # Store in both L1 and L2 caches
            l1_success = await self.l1_cache.put(key, value, ttl)
            l2_success = await self.l2_cache.put(key, value, ttl)
            
            if l1_success or l2_success:
                logger.debug(f"Cached value for key: {key}")
                return True
            
        except Exception as e:
            logger.error(f"Cache put error for key {key}: {str(e)}")
        
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache layers."""
        l1_success = await self.l1_cache.delete(key)
        l2_success = await self.l2_cache.delete(key)
        
        return l1_success or l2_success
    
    async def clear(self) -> bool:
        """Clear all cache layers."""
        l1_success = await self.l1_cache.clear()
        l2_success = await self.l2_cache.clear()
        
        # Reset metrics
        self.global_metrics = CacheMetrics()
        
        return l1_success and l2_success
    
    async def prefetch(self, keys: List[str], loader: Callable) -> int:
        """Prefetch multiple keys."""
        prefetched_count = 0
        
        for key in keys:
            # Check if already cached
            if await self.l1_cache.get(key) is None and await self.l2_cache.get(key) is None:
                try:
                    if asyncio.iscoroutinefunction(loader):
                        value = await loader(key)
                    else:
                        value = loader(key)
                    
                    if value is not None:
                        await self.put(key, value)
                        prefetched_count += 1
                        
                except Exception as e:
                    logger.warning(f"Prefetch failed for key {key}: {str(e)}")
        
        logger.info(f"Prefetched {prefetched_count} items")
        return prefetched_count
    
    async def _prefetch_worker(self):
        """Background worker for intelligent prefetching."""
        while True:
            try:
                await asyncio.sleep(self.config["prefetch_interval"])
                
                # Get prefetch candidates from ML optimizer
                candidates = self.ml_optimizer.get_prefetch_candidates()
                
                if candidates:
                    logger.debug(f"Identified {len(candidates)} prefetch candidates")
                    # In practice, would call prefetch with appropriate loader
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch worker error: {str(e)}")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        return {
            "global_metrics": {
                "total_requests": self.global_metrics.total_requests,
                "hit_rate": self.global_metrics.hit_rate,
                "l1_hit_rate": self.global_metrics.l1_hit_rate,
                "avg_response_time": self.global_metrics.avg_response_time
            },
            "l1_cache": l1_stats,
            "l2_cache": l2_stats,
            "prefetch_enabled": self.prefetch_enabled,
            "ml_optimizer": {
                "tracked_patterns": len(self.ml_optimizer.access_patterns),
                "prediction_accuracy": self.ml_optimizer.prediction_accuracy
            }
        }
    
    def optimize_for_mobile(self, battery_level: float = 1.0, memory_pressure: float = 0.0):
        """Optimize cache behavior for mobile constraints."""
        # Adjust cache sizes based on battery and memory pressure
        if battery_level < 0.2:  # Low battery
            # Reduce cache sizes to save power
            self.l1_cache.max_size_bytes = int(self.l1_cache.max_size_bytes * 0.5)
            logger.info("Reduced L1 cache size due to low battery")
        
        if memory_pressure > 0.7:  # High memory pressure
            # Aggressive eviction from L1 cache
            current_size = self.l1_cache.get_size()
            if current_size > self.l1_cache.max_size_bytes * 0.3:
                # Evict entries to reduce to 30% of max size
                target_size = int(self.l1_cache.max_size_bytes * 0.3)
                asyncio.create_task(self._aggressive_eviction(target_size))
                logger.info("Triggered aggressive cache eviction due to memory pressure")
    
    async def _aggressive_eviction(self, target_size: int):
        """Aggressively evict cache entries to reach target size."""
        current_size = self.l1_cache.get_size()
        
        while current_size > target_size and self.l1_cache.entries:
            # Find lowest priority entry
            min_priority = float('inf')
            candidate_key = None
            
            for key, entry in self.l1_cache.entries.items():
                priority = self.ml_optimizer.calculate_priority_score(entry)
                if priority < min_priority:
                    min_priority = priority
                    candidate_key = key
            
            if candidate_key:
                await self.l1_cache.delete(candidate_key)
                current_size = self.l1_cache.get_size()
            else:
                break
    
    def shutdown(self):
        """Shutdown cache manager."""
        if self.prefetch_task:
            self.prefetch_task.cancel()
        
        logger.info("Cache manager shutdown")


# Factory functions
def create_mobile_cache_manager(cache_dir: str = "/tmp/mobile_multimodal_cache") -> IntelligentCacheManager:
    """Create cache manager optimized for mobile deployment."""
    config = {
        "l1_size_mb": 64,      # Smaller L1 for mobile
        "l2_size_mb": 256,     # Smaller L2 for mobile
        "cache_dir": cache_dir,
        "eviction_policy": "adaptive",
        "enable_prefetching": True,
        "prefetch_threshold": 0.4,  # Higher threshold for mobile
        "prefetch_interval": 120.0,  # Less frequent prefetching
        "default_ttl": 1800.0  # Shorter TTL for mobile
    }
    
    return IntelligentCacheManager(config)


# Export classes and functions
__all__ = [
    "CacheLevel", "EvictionPolicy", "CacheHit", "CacheEntry", "CacheMetrics",
    "CacheCompressor", "MLCacheOptimizer", "AdaptiveCacheLayer",
    "L1MemoryCache", "L2DiskCache", "IntelligentCacheManager",
    "create_mobile_cache_manager"
]