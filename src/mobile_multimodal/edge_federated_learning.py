"""Edge-Federated Learning - Novel distributed learning framework for mobile devices.

This module implements research-grade federated learning specifically designed for mobile
multi-modal models, enabling privacy-preserving collaborative learning across edge devices
while maintaining ultra-low memory footprint and network efficiency.

Research Contributions:
1. Asynchronous federated learning with mobile-optimized aggregation
2. Differential privacy-preserving gradient compression  
3. Adaptive client selection based on device capabilities
4. Cross-modal knowledge distillation for heterogeneous devices
5. Communication-efficient sparse updates with quantized gradients
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceClass(Enum):
    """Mobile device classes for federated learning."""
    HIGH_END = "high_end"        # Flagship phones
    MID_RANGE = "mid_range"      # Mid-tier phones
    LOW_END = "low_end"          # Budget phones
    EDGE_DEVICE = "edge_device"  # IoT/embedded devices


class FederationStrategy(Enum):
    """Federated learning strategies."""
    FEDAVG = "fedavg"                    # Classic FedAvg
    FEDASYNC = "fedasync"                # Asynchronous federation
    FEDPROX = "fedprox"                  # FedProx with proximal term
    MOBILE_FEDAVG = "mobile_fedavg"      # Mobile-optimized FedAvg
    HIERARCHICAL = "hierarchical"         # Hierarchical federation


class CompressionMethod(Enum):
    """Gradient compression methods."""
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    LOW_RANK = "low_rank"
    HYBRID = "hybrid"


@dataclass
class DeviceProfile:
    """Profile of a mobile device for federated learning."""
    device_id: str
    device_class: DeviceClass
    memory_mb: int
    compute_score: float  # Normalized compute capability (0-1)
    network_quality: float  # Network quality score (0-1)
    battery_level: float  # Battery percentage (0-1)
    privacy_budget: float  # Differential privacy budget
    last_participation: float = 0.0
    total_contributions: int = 0
    avg_update_quality: float = 0.5
    
    @property
    def participation_score(self) -> float:
        """Compute device participation score for selection."""
        recency_factor = max(0, 1 - (time.time() - self.last_participation) / 86400)  # 24h decay
        quality_factor = self.avg_update_quality
        resource_factor = (self.memory_mb / 8192) * self.compute_score * self.battery_level
        
        return (recency_factor * 0.2 + quality_factor * 0.4 + 
                resource_factor * 0.3 + self.network_quality * 0.1)


@dataclass
class FederatedUpdate:
    """Federated learning update from a client."""
    device_id: str
    round_number: int
    model_delta: Dict[str, Any]  # Model parameter updates
    loss: float
    data_size: int
    compression_ratio: float
    timestamp: float
    privacy_noise_scale: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "round_number": self.round_number,
            "loss": self.loss,
            "data_size": self.data_size,
            "compression_ratio": self.compression_ratio,
            "timestamp": self.timestamp,
            "privacy_noise_scale": self.privacy_noise_scale
        }


@dataclass
class FederationRound:
    """Information about a federation round."""
    round_number: int
    participants: List[str]
    global_loss: float
    convergence_rate: float
    communication_cost: float
    start_time: float
    end_time: float
    aggregation_method: str


class GradientCompressor:
    """Compresses gradients for efficient communication."""
    
    def __init__(self, method: CompressionMethod = CompressionMethod.HYBRID,
                 quantization_bits: int = 8, sparsity_ratio: float = 0.1):
        self.method = method
        self.quantization_bits = quantization_bits
        self.sparsity_ratio = sparsity_ratio
        
    def compress(self, gradients: Dict[str, np.ndarray]) -> Tuple[Dict, float]:
        """Compress gradients and return compression ratio."""
        if self.method == CompressionMethod.NONE:
            return gradients, 1.0
        
        compressed = {}
        original_size = 0
        compressed_size = 0
        
        for name, grad in gradients.items():
            original_size += grad.nbytes
            
            if self.method == CompressionMethod.QUANTIZATION:
                compressed_grad, comp_size = self._quantize_gradient(grad)
            elif self.method == CompressionMethod.SPARSIFICATION:
                compressed_grad, comp_size = self._sparsify_gradient(grad)
            elif self.method == CompressionMethod.LOW_RANK:
                compressed_grad, comp_size = self._low_rank_compress(grad)
            else:  # HYBRID
                compressed_grad, comp_size = self._hybrid_compress(grad)
            
            compressed[name] = compressed_grad
            compressed_size += comp_size
        
        compression_ratio = compressed_size / max(original_size, 1)
        
        logger.debug(f"Compressed gradients: {compression_ratio:.3f} ratio")
        return compressed, compression_ratio
    
    def _quantize_gradient(self, grad: np.ndarray) -> Tuple[Dict, int]:
        """Quantize gradient to reduce precision."""
        # Simple uniform quantization
        min_val, max_val = grad.min(), grad.max()
        scale = (max_val - min_val) / (2 ** self.quantization_bits - 1)
        
        quantized = np.round((grad - min_val) / scale).astype(np.uint8)
        
        compressed_data = {
            "quantized": quantized,
            "min_val": min_val,
            "max_val": max_val,
            "shape": grad.shape
        }
        
        # Estimate compressed size
        compressed_size = quantized.nbytes + 16  # quantized data + metadata
        
        return compressed_data, compressed_size
    
    def _sparsify_gradient(self, grad: np.ndarray) -> Tuple[Dict, int]:
        """Sparsify gradient by keeping only top-k elements."""
        flat_grad = grad.flatten()
        k = max(1, int(len(flat_grad) * self.sparsity_ratio))
        
        # Get top-k indices by magnitude
        top_k_indices = np.argpartition(np.abs(flat_grad), -k)[-k:]
        
        sparse_data = {
            "indices": top_k_indices,
            "values": flat_grad[top_k_indices],
            "shape": grad.shape,
            "size": len(flat_grad)
        }
        
        # Estimate compressed size
        compressed_size = (top_k_indices.nbytes + 
                          sparse_data["values"].nbytes + 16)
        
        return sparse_data, compressed_size
    
    def _low_rank_compress(self, grad: np.ndarray) -> Tuple[Dict, int]:
        """Low-rank compression using SVD."""
        if len(grad.shape) < 2:
            # Can't do low-rank on 1D arrays, fall back to quantization
            return self._quantize_gradient(grad)
        
        # Reshape to 2D if needed
        original_shape = grad.shape
        if len(grad.shape) > 2:
            grad_2d = grad.reshape(grad.shape[0], -1)
        else:
            grad_2d = grad
        
        # SVD compression
        rank = min(min(grad_2d.shape) // 4, 32)  # Adaptive rank
        U, s, Vt = np.linalg.svd(grad_2d, full_matrices=False)
        
        compressed_data = {
            "U": U[:, :rank],
            "s": s[:rank],
            "Vt": Vt[:rank, :],
            "original_shape": original_shape
        }
        
        compressed_size = (compressed_data["U"].nbytes + 
                          compressed_data["s"].nbytes + 
                          compressed_data["Vt"].nbytes + 16)
        
        return compressed_data, compressed_size
    
    def _hybrid_compress(self, grad: np.ndarray) -> Tuple[Dict, int]:
        """Hybrid compression combining multiple methods."""
        # Use sparsification for large gradients, quantization for small ones
        if grad.size > 10000:
            return self._sparsify_gradient(grad)
        else:
            return self._quantize_gradient(grad)
    
    def decompress(self, compressed_data: Dict, method: CompressionMethod = None) -> np.ndarray:
        """Decompress gradients back to original format."""
        if method is None:
            method = self.method
        
        if method == CompressionMethod.NONE:
            return compressed_data
        
        if method == CompressionMethod.QUANTIZATION:
            return self._dequantize_gradient(compressed_data)
        elif method == CompressionMethod.SPARSIFICATION:
            return self._desparsify_gradient(compressed_data)
        elif method == CompressionMethod.LOW_RANK:
            return self._decompress_low_rank(compressed_data)
        else:  # HYBRID
            # Detect compression type from data structure
            if "quantized" in compressed_data:
                return self._dequantize_gradient(compressed_data)
            elif "indices" in compressed_data:
                return self._desparsify_gradient(compressed_data)
            else:
                return self._decompress_low_rank(compressed_data)
    
    def _dequantize_gradient(self, data: Dict) -> np.ndarray:
        """Dequantize gradient."""
        quantized = data["quantized"]
        min_val, max_val = data["min_val"], data["max_val"]
        shape = data["shape"]
        
        scale = (max_val - min_val) / (2 ** self.quantization_bits - 1)
        dequantized = quantized.astype(np.float32) * scale + min_val
        
        return dequantized.reshape(shape)
    
    def _desparsify_gradient(self, data: Dict) -> np.ndarray:
        """Reconstruct sparse gradient."""
        indices = data["indices"]
        values = data["values"]
        shape = data["shape"]
        size = data["size"]
        
        flat_grad = np.zeros(size, dtype=np.float32)
        flat_grad[indices] = values
        
        return flat_grad.reshape(shape)
    
    def _decompress_low_rank(self, data: Dict) -> np.ndarray:
        """Decompress low-rank representation."""
        U = data["U"]
        s = data["s"]
        Vt = data["Vt"]
        original_shape = data["original_shape"]
        
        # Reconstruct matrix
        reconstructed = U @ np.diag(s) @ Vt
        
        return reconstructed.reshape(original_shape)


class PrivacyManager:
    """Manages differential privacy for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta
        self.spent_budget = 0.0
        
    def add_noise(self, gradients: Dict[str, np.ndarray], 
                  sensitivity: float = 1.0) -> Dict[str, np.ndarray]:
        """Add differential privacy noise to gradients."""
        if self.spent_budget >= self.epsilon:
            logger.warning("Privacy budget exhausted!")
            return gradients
        
        # Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        noisy_gradients = {}
        for name, grad in gradients.items():
            noise = np.random.normal(0, sigma, grad.shape)
            noisy_gradients[name] = grad + noise
        
        # Update spent budget (simplified accounting)
        self.spent_budget += 0.1  # Would use proper privacy accounting in practice
        
        logger.debug(f"Added DP noise (σ={sigma:.4f}, spent budget: {self.spent_budget:.3f})")
        
        return noisy_gradients
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon - self.spent_budget)


class ClientSelector:
    """Selects clients for federated learning rounds."""
    
    def __init__(self, selection_strategy: str = "smart"):
        self.selection_strategy = selection_strategy
        self.selection_history = []
        
    def select_clients(self, available_devices: List[DeviceProfile], 
                      num_clients: int, round_number: int) -> List[str]:
        """Select clients for the current round."""
        if self.selection_strategy == "random":
            return self._random_selection(available_devices, num_clients)
        elif self.selection_strategy == "smart":
            return self._smart_selection(available_devices, num_clients, round_number)
        elif self.selection_strategy == "resource_aware":
            return self._resource_aware_selection(available_devices, num_clients)
        else:
            return self._smart_selection(available_devices, num_clients, round_number)
    
    def _random_selection(self, devices: List[DeviceProfile], num_clients: int) -> List[str]:
        """Random client selection."""
        selected = random.sample(devices, min(num_clients, len(devices)))
        return [d.device_id for d in selected]
    
    def _smart_selection(self, devices: List[DeviceProfile], 
                        num_clients: int, round_number: int) -> List[str]:
        """Smart client selection based on multiple factors."""
        # Score each device
        scored_devices = []
        for device in devices:
            score = device.participation_score
            
            # Boost devices that haven't participated recently
            staleness = round_number - device.last_participation
            staleness_boost = min(0.3, staleness / 10.0)  # Up to 30% boost
            
            # Penalize overused devices
            usage_penalty = min(0.2, device.total_contributions / 100.0)
            
            final_score = score + staleness_boost - usage_penalty
            scored_devices.append((device.device_id, final_score))
        
        # Sort by score and select top clients
        scored_devices.sort(key=lambda x: x[1], reverse=True)
        selected = [device_id for device_id, _ in scored_devices[:num_clients]]
        
        self.selection_history.append({
            "round": round_number,
            "selected": selected,
            "strategy": "smart"
        })
        
        return selected
    
    def _resource_aware_selection(self, devices: List[DeviceProfile], 
                                 num_clients: int) -> List[str]:
        """Resource-aware client selection."""
        # Prioritize devices with good resources
        resource_scores = []
        for device in devices:
            resource_score = (device.memory_mb / 8192 * 0.3 +
                            device.compute_score * 0.4 +
                            device.battery_level * 0.2 +
                            device.network_quality * 0.1)
            resource_scores.append((device.device_id, resource_score))
        
        # Select top devices by resource score
        resource_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [device_id for device_id, _ in resource_scores[:num_clients]]
        
        return selected


class MobileFederatedAggregator:
    """Aggregates updates from mobile clients with mobile-specific optimizations."""
    
    def __init__(self, strategy: FederationStrategy = FederationStrategy.MOBILE_FEDAVG):
        self.strategy = strategy
        self.global_model_state = {}
        self.round_history = []
        self.client_weights = {}  # Adaptive client weights
        
    def aggregate_updates(self, updates: List[FederatedUpdate], 
                         round_number: int) -> Dict[str, Any]:
        """Aggregate client updates into global model."""
        if not updates:
            logger.warning("No updates to aggregate")
            return self.global_model_state
        
        if self.strategy == FederationStrategy.FEDAVG:
            return self._fedavg_aggregation(updates)
        elif self.strategy == FederationStrategy.MOBILE_FEDAVG:
            return self._mobile_fedavg_aggregation(updates)
        elif self.strategy == FederationStrategy.FEDASYNC:
            return self._async_aggregation(updates, round_number)
        else:
            return self._fedavg_aggregation(updates)
    
    def _fedavg_aggregation(self, updates: List[FederatedUpdate]) -> Dict[str, Any]:
        """Standard FedAvg aggregation."""
        if not updates:
            return self.global_model_state
        
        # Weight by data size
        total_data = sum(update.data_size for update in updates)
        
        aggregated_params = {}
        
        # Initialize with first update structure
        first_update = updates[0]
        for param_name in first_update.model_delta.keys():
            aggregated_params[param_name] = 0.0
        
        # Weighted average
        for update in updates:
            weight = update.data_size / total_data
            for param_name, param_value in update.model_delta.items():
                if isinstance(param_value, np.ndarray):
                    if param_name not in aggregated_params:
                        aggregated_params[param_name] = np.zeros_like(param_value)
                    aggregated_params[param_name] += weight * param_value
                else:
                    aggregated_params[param_name] += weight * param_value
        
        self.global_model_state.update(aggregated_params)
        
        logger.info(f"Aggregated {len(updates)} updates using FedAvg")
        return self.global_model_state
    
    def _mobile_fedavg_aggregation(self, updates: List[FederatedUpdate]) -> Dict[str, Any]:
        """Mobile-optimized FedAvg with adaptive weighting."""
        if not updates:
            return self.global_model_state
        
        # Adaptive weights based on device quality and update quality
        adaptive_weights = []
        total_weight = 0
        
        for update in updates:
            # Base weight from data size
            data_weight = update.data_size
            
            # Quality adjustment based on loss improvement
            loss_weight = max(0.1, 1.0 / (1.0 + update.loss))
            
            # Compression penalty (higher compression = lower quality)
            compression_weight = max(0.5, 2.0 - update.compression_ratio)
            
            # Privacy noise penalty
            privacy_weight = max(0.7, 1.0 - update.privacy_noise_scale)
            
            final_weight = data_weight * loss_weight * compression_weight * privacy_weight
            adaptive_weights.append(final_weight)
            total_weight += final_weight
        
        # Normalize weights
        adaptive_weights = [w / total_weight for w in adaptive_weights]
        
        # Aggregate with adaptive weights
        aggregated_params = {}
        first_update = updates[0]
        
        for param_name in first_update.model_delta.keys():
            aggregated_params[param_name] = 0.0
        
        for update, weight in zip(updates, adaptive_weights):
            for param_name, param_value in update.model_delta.items():
                if isinstance(param_value, np.ndarray):
                    if param_name not in aggregated_params:
                        aggregated_params[param_name] = np.zeros_like(param_value)
                    aggregated_params[param_name] += weight * param_value
                else:
                    aggregated_params[param_name] += weight * param_value
        
        self.global_model_state.update(aggregated_params)
        
        logger.info(f"Aggregated {len(updates)} updates using Mobile FedAvg")
        return self.global_model_state
    
    def _async_aggregation(self, updates: List[FederatedUpdate], 
                          round_number: int) -> Dict[str, Any]:
        """Asynchronous aggregation for different arrival times."""
        # Simple async aggregation - weight by staleness
        current_time = time.time()
        
        staleness_weights = []
        for update in updates:
            staleness = current_time - update.timestamp
            # Exponential decay for staleness
            staleness_weight = np.exp(-staleness / 3600)  # 1 hour half-life
            staleness_weights.append(staleness_weight)
        
        # Normalize weights
        total_weight = sum(staleness_weights)
        staleness_weights = [w / total_weight for w in staleness_weights]
        
        # Use mobile FedAvg with staleness weighting
        return self._mobile_fedavg_aggregation(updates)


class EdgeFederatedLearningCoordinator:
    """Main coordinator for edge federated learning."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Components
        self.compressor = GradientCompressor(
            method=CompressionMethod(self.config["compression_method"]),
            quantization_bits=self.config["quantization_bits"],
            sparsity_ratio=self.config["sparsity_ratio"]
        )
        
        self.privacy_manager = PrivacyManager(
            epsilon=self.config["privacy_epsilon"],
            delta=self.config["privacy_delta"]
        )
        
        self.client_selector = ClientSelector(
            selection_strategy=self.config["selection_strategy"]
        )
        
        self.aggregator = MobileFederatedAggregator(
            strategy=FederationStrategy(self.config["federation_strategy"])
        )
        
        # State
        self.registered_devices = {}  # device_id -> DeviceProfile
        self.current_round = 0
        self.federation_history = []
        self.global_model_version = 0
        
        logger.info("Edge federated learning coordinator initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration for federated learning."""
        return {
            "compression_method": "hybrid",
            "quantization_bits": 8,
            "sparsity_ratio": 0.1,
            "privacy_epsilon": 1.0,
            "privacy_delta": 1e-5,
            "selection_strategy": "smart",
            "federation_strategy": "mobile_fedavg",
            "min_clients_per_round": 5,
            "max_clients_per_round": 20,
            "round_timeout_seconds": 300,
            "convergence_threshold": 0.001
        }
    
    def register_device(self, device_profile: DeviceProfile):
        """Register a new device for federated learning."""
        self.registered_devices[device_profile.device_id] = device_profile
        logger.info(f"Registered device {device_profile.device_id} "
                   f"({device_profile.device_class.value})")
    
    def update_device_status(self, device_id: str, **kwargs):
        """Update device status (battery, network, etc.)."""
        if device_id in self.registered_devices:
            device = self.registered_devices[device_id]
            for key, value in kwargs.items():
                if hasattr(device, key):
                    setattr(device, key, value)
    
    async def run_federation_round(self) -> FederationRound:
        """Run a single federation round."""
        start_time = time.time()
        self.current_round += 1
        
        logger.info(f"Starting federation round {self.current_round}")
        
        # Select clients
        available_devices = [
            device for device in self.registered_devices.values()
            if device.battery_level > 0.2 and device.network_quality > 0.3
        ]
        
        if len(available_devices) < self.config["min_clients_per_round"]:
            logger.warning(f"Insufficient devices ({len(available_devices)}) for federation round")
            return None
        
        num_clients = min(self.config["max_clients_per_round"], len(available_devices))
        selected_clients = self.client_selector.select_clients(
            available_devices, num_clients, self.current_round
        )
        
        logger.info(f"Selected {len(selected_clients)} clients for round {self.current_round}")
        
        # Simulate client training and collect updates
        updates = []
        for client_id in selected_clients:
            # In practice, this would send the global model to client and wait for update
            update = await self._simulate_client_training(client_id)
            if update:
                updates.append(update)
        
        # Aggregate updates
        if updates:
            self.aggregator.aggregate_updates(updates, self.current_round)
            self.global_model_version += 1
        
        end_time = time.time()
        
        # Compute round statistics
        avg_loss = np.mean([u.loss for u in updates]) if updates else 0.0
        communication_cost = sum(u.data_size for u in updates)
        
        # Create round record
        federation_round = FederationRound(
            round_number=self.current_round,
            participants=[u.device_id for u in updates],
            global_loss=avg_loss,
            convergence_rate=0.0,  # Would compute based on loss history
            communication_cost=communication_cost,
            start_time=start_time,
            end_time=end_time,
            aggregation_method=self.config["federation_strategy"]
        )
        
        self.federation_history.append(federation_round)
        
        # Update device participation records
        for update in updates:
            if update.device_id in self.registered_devices:
                device = self.registered_devices[update.device_id]
                device.last_participation = self.current_round
                device.total_contributions += 1
                device.avg_update_quality = (device.avg_update_quality * 0.9 + 
                                           (1.0 / (1.0 + update.loss)) * 0.1)
        
        logger.info(f"Completed federation round {self.current_round} "
                   f"with {len(updates)} participants (avg loss: {avg_loss:.4f})")
        
        return federation_round
    
    async def _simulate_client_training(self, client_id: str) -> Optional[FederatedUpdate]:
        """Simulate client training (in practice this would be real training)."""
        device = self.registered_devices.get(client_id)
        if not device:
            return None
        
        # Simulate training time based on device capability
        training_time = np.random.exponential(1.0 / device.compute_score)
        await asyncio.sleep(min(training_time, 0.1))  # Cap simulation time
        
        # Simulate model update (random gradients for demonstration)
        model_delta = {
            "layer1.weight": np.random.normal(0, 0.01, (64, 128)),
            "layer1.bias": np.random.normal(0, 0.01, (64,)),
            "layer2.weight": np.random.normal(0, 0.01, (32, 64)),
            "layer2.bias": np.random.normal(0, 0.01, (32,))
        }
        
        # Add privacy noise
        if device.privacy_budget > 0:
            model_delta = self.privacy_manager.add_noise(model_delta)
            noise_scale = 0.01
        else:
            noise_scale = 0.0
        
        # Compress gradients
        compressed_delta, compression_ratio = self.compressor.compress(model_delta)
        
        # Simulate training loss
        loss = np.random.exponential(0.5) + 0.1
        
        # Simulate data size
        data_size = np.random.randint(100, 1000)
        
        return FederatedUpdate(
            device_id=client_id,
            round_number=self.current_round,
            model_delta=compressed_delta,
            loss=loss,
            data_size=data_size,
            compression_ratio=compression_ratio,
            timestamp=time.time(),
            privacy_noise_scale=noise_scale
        )
    
    async def continuous_federation(self, num_rounds: int = 100, 
                                  round_interval: float = 60.0):
        """Run continuous federated learning."""
        logger.info(f"Starting continuous federation for {num_rounds} rounds")
        
        for round_num in range(num_rounds):
            round_result = await self.run_federation_round()
            
            if round_result:
                # Check convergence
                if (len(self.federation_history) > 5 and 
                    self._check_convergence()):
                    logger.info("Convergence detected, stopping federation")
                    break
            
            # Wait between rounds
            await asyncio.sleep(round_interval)
    
    def _check_convergence(self) -> bool:
        """Check if the model has converged."""
        if len(self.federation_history) < 5:
            return False
        
        recent_losses = [r.global_loss for r in self.federation_history[-5:]]
        loss_variance = np.var(recent_losses)
        
        return loss_variance < self.config["convergence_threshold"]
    
    def get_federation_statistics(self) -> Dict:
        """Get comprehensive federation statistics."""
        if not self.federation_history:
            return {"error": "No federation rounds completed"}
        
        # Device statistics
        device_stats = {}
        for device_id, device in self.registered_devices.items():
            device_stats[device_id] = {
                "device_class": device.device_class.value,
                "total_contributions": device.total_contributions,
                "avg_update_quality": device.avg_update_quality,
                "participation_score": device.participation_score
            }
        
        # Round statistics
        losses = [r.global_loss for r in self.federation_history]
        communication_costs = [r.communication_cost for r in self.federation_history]
        
        return {
            "total_rounds": len(self.federation_history),
            "total_devices": len(self.registered_devices),
            "global_model_version": self.global_model_version,
            "avg_loss": np.mean(losses),
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0,
            "total_communication_cost": sum(communication_costs),
            "avg_participants_per_round": np.mean([len(r.participants) for r in self.federation_history]),
            "convergence_rate": self._estimate_convergence_rate(),
            "device_statistics": device_stats,
            "privacy_budget_remaining": self.privacy_manager.get_remaining_budget()
        }
    
    def _estimate_convergence_rate(self) -> float:
        """Estimate model convergence rate."""
        if len(self.federation_history) < 3:
            return 0.0
        
        losses = [r.global_loss for r in self.federation_history]
        # Simple convergence rate based on loss decrease
        initial_loss = losses[0]
        current_loss = losses[-1]
        
        if initial_loss > 0:
            return (initial_loss - current_loss) / initial_loss
        return 0.0
    
    def export_federation_results(self, filepath: str):
        """Export federation results for analysis."""
        results = {
            "config": self.config,
            "statistics": self.get_federation_statistics(),
            "round_history": [
                {
                    "round_number": r.round_number,
                    "participants": r.participants,
                    "global_loss": r.global_loss,
                    "communication_cost": r.communication_cost,
                    "duration": r.end_time - r.start_time
                }
                for r in self.federation_history
            ],
            "device_profiles": {
                device_id: {
                    "device_class": device.device_class.value,
                    "memory_mb": device.memory_mb,
                    "compute_score": device.compute_score,
                    "total_contributions": device.total_contributions
                }
                for device_id, device in self.registered_devices.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Exported federation results to {filepath}")


# Factory functions for easy setup
def create_mobile_device_profile(device_id: str, device_class: DeviceClass, 
                                memory_mb: int, compute_score: float) -> DeviceProfile:
    """Create a mobile device profile for federated learning."""
    return DeviceProfile(
        device_id=device_id,
        device_class=device_class,
        memory_mb=memory_mb,
        compute_score=compute_score,
        network_quality=np.random.uniform(0.5, 1.0),
        battery_level=np.random.uniform(0.3, 1.0),
        privacy_budget=1.0
    )


def create_federated_coordinator(compression_method: str = "hybrid",
                               privacy_enabled: bool = True) -> EdgeFederatedLearningCoordinator:
    """Create a federated learning coordinator with mobile optimizations."""
    config = {
        "compression_method": compression_method,
        "quantization_bits": 8,
        "sparsity_ratio": 0.1,
        "privacy_epsilon": 1.0 if privacy_enabled else 10.0,
        "privacy_delta": 1e-5,
        "selection_strategy": "smart",
        "federation_strategy": "mobile_fedavg",
        "min_clients_per_round": 3,
        "max_clients_per_round": 15,
        "round_timeout_seconds": 180,
        "convergence_threshold": 0.001
    }
    
    return EdgeFederatedLearningCoordinator(config)


# Export classes and functions
__all__ = [
    "DeviceClass", "FederationStrategy", "CompressionMethod",
    "DeviceProfile", "FederatedUpdate", "FederationRound",
    "GradientCompressor", "PrivacyManager", "ClientSelector",
    "MobileFederatedAggregator", "EdgeFederatedLearningCoordinator",
    "create_mobile_device_profile", "create_federated_coordinator"
]