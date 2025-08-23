"""Advanced Distributed Scaling for Mobile Multi-Modal AI Systems.

Comprehensive scaling framework with distributed inference, auto-scaling,
load balancing, and global deployment orchestration.
"""

import asyncio
import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid
import hashlib
from datetime import datetime, timedelta
import concurrent.futures
import queue
from enum import Enum
import multiprocessing as mp

# Distributed computing libraries
try:
    import redis
    import grpc
    from concurrent import futures as grpc_futures
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    # Mock implementations for environments without distributed libraries
    DISTRIBUTED_AVAILABLE = False
    redis = None
    grpc = None
    
    class MockRedis:
        def __init__(self, *args, **kwargs):
            self._data = {}
        
        def get(self, key):
            return self._data.get(key)
        
        def set(self, key, value, ex=None):
            self._data[key] = value
            return True
        
        def lpush(self, key, value):
            if key not in self._data:
                self._data[key] = []
            self._data[key].insert(0, value)
        
        def brpop(self, keys, timeout=None):
            for key in keys:
                if key in self._data and self._data[key]:
                    return (key, self._data[key].pop())
            return None
        
        def publish(self, channel, message):
            pass

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"           # Scale based on current load
    PREDICTIVE = "predictive"       # Scale based on predicted load
    SCHEDULED = "scheduled"         # Scale based on schedule
    HYBRID = "hybrid"              # Combination of strategies

class NodeStatus(Enum):
    """Node status in distributed system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

@dataclass
class WorkerNode:
    """Distributed worker node configuration."""
    node_id: str
    hostname: str
    port: int
    capabilities: List[str]
    max_concurrent_tasks: int
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    status: NodeStatus = NodeStatus.OFFLINE
    last_heartbeat: float = 0
    current_tasks: int = 0
    total_tasks_completed: int = 0
    average_task_time: float = 0
    
@dataclass
class InferenceTask:
    """Distributed inference task."""
    task_id: str
    model_name: str
    task_type: str  # caption, ocr, vqa, etc.
    input_data: Any
    priority: int = 1
    timeout: float = 30.0
    created_at: float = 0
    assigned_to: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    timestamp: float
    total_nodes: int
    healthy_nodes: int
    queue_length: int
    average_response_time: float
    cpu_utilization: float
    memory_utilization: float
    requests_per_second: float
    error_rate: float

class DistributedScaling:
    """Advanced distributed scaling and orchestration system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize distributed scaling system.
        
        Args:
            config: Scaling configuration
        """
        self.config = config or {}
        
        # Redis configuration for distributed coordination
        self.redis_host = self.config.get("redis_host", "localhost")
        self.redis_port = self.config.get("redis_port", 6379)
        self.redis_db = self.config.get("redis_db", 0)
        
        # Initialize Redis connection
        if DISTRIBUTED_AVAILABLE and redis:
            self.redis_client = redis.Redis(
                host=self.redis_host, 
                port=self.redis_port, 
                db=self.redis_db,
                decode_responses=True
            )
        else:
            self.redis_client = MockRedis()
        
        # Node management
        self.worker_nodes = {}
        self.node_id = str(uuid.uuid4())
        self.is_coordinator = False
        
        # Task queue and processing
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=10000)
        
        # Auto-scaling configuration
        self.scaling_strategy = ScalingStrategy(self.config.get("scaling_strategy", "reactive"))
        self.min_nodes = self.config.get("min_nodes", 1)
        self.max_nodes = self.config.get("max_nodes", 10)
        self.target_cpu_utilization = self.config.get("target_cpu_utilization", 70.0)
        self.target_queue_length = self.config.get("target_queue_length", 10)
        self.scale_up_threshold = self.config.get("scale_up_threshold", 0.8)
        self.scale_down_threshold = self.config.get("scale_down_threshold", 0.3)
        
        # Load balancing
        self.load_balancer = LoadBalancer(self)
        
        # Metrics collection
        self.metrics_history = deque(maxlen=1000)
        self.performance_predictors = {}
        
        # Threading and async management
        self._shutdown_event = threading.Event()
        self._threads = []
        self._async_loop = None
        self._async_tasks = []
        
        # Health monitoring
        self.health_check_interval = self.config.get("health_check_interval", 30)
        self.heartbeat_interval = self.config.get("heartbeat_interval", 10)
        
        logger.info(f"Distributed scaling system initialized with node ID: {self.node_id}")
    
    def start_distributed_system(self, is_coordinator: bool = False):
        """Start the distributed system.
        
        Args:
            is_coordinator: Whether this node is the coordinator
        """
        self.is_coordinator = is_coordinator
        
        # Register this node
        self._register_node()
        
        # Start core threads
        self._start_heartbeat_thread()
        self._start_health_monitor_thread()
        self._start_task_processor_thread()
        
        if self.is_coordinator:
            self._start_coordinator_threads()
        
        # Start async event loop
        self._start_async_loop()
        
        logger.info(f"Distributed system started (coordinator: {is_coordinator})")
    
    def stop_distributed_system(self):
        """Stop the distributed system."""
        self._shutdown_event.set()
        
        # Stop async tasks
        for task in self._async_tasks:
            task.cancel()
        
        if self._async_loop:
            self._async_loop.stop()
        
        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=5)
        
        # Unregister node
        self._unregister_node()
        
        logger.info("Distributed system stopped")
    
    def submit_inference_task(self, model_name: str, task_type: str, 
                            input_data: Any, priority: int = 1, 
                            timeout: float = 30.0) -> str:
        """Submit inference task to distributed system.
        
        Args:
            model_name: Name of the model to use
            task_type: Type of inference task
            input_data: Input data for inference
            priority: Task priority (lower number = higher priority)
            timeout: Task timeout in seconds
            
        Returns:
            Task ID
        """
        task = InferenceTask(
            task_id=str(uuid.uuid4()),
            model_name=model_name,
            task_type=task_type,
            input_data=input_data,
            priority=priority,
            timeout=timeout,
            created_at=time.time()
        )
        
        # Serialize task for distributed queue
        task_data = {
            "task_id": task.task_id,
            "model_name": task.model_name,
            "task_type": task.task_type,
            "input_data": self._serialize_input_data(task.input_data),
            "priority": task.priority,
            "timeout": task.timeout,
            "created_at": task.created_at
        }
        
        # Add to distributed queue
        self.redis_client.lpush("inference_tasks", json.dumps(task_data))
        
        # Store task locally for tracking
        self.active_tasks[task.task_id] = task
        
        logger.debug(f"Submitted inference task: {task.task_id}")
        return task.task_id
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result of inference task.
        
        Args:
            task_id: Task identifier
            timeout: Optional timeout for waiting
            
        Returns:
            Task result or None if not ready/failed
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            # Check completed tasks
            for completed_task in self.completed_tasks:
                if completed_task.task_id == task_id:
                    if completed_task.error:
                        raise Exception(f"Task failed: {completed_task.error}")
                    return completed_task.result
            
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.result is not None:
                    return task.result
                elif task.error is not None:
                    raise Exception(f"Task failed: {task.error}")
            
            # Check Redis for result
            result_key = f"task_result:{task_id}"
            result_data = self.redis_client.get(result_key)
            if result_data:
                result_info = json.loads(result_data)
                if result_info.get("error"):
                    raise Exception(f"Task failed: {result_info['error']}")
                return self._deserialize_result(result_info.get("result"))
            
            time.sleep(0.1)  # Short sleep to avoid busy waiting
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get distributed system status.
        
        Returns:
            System status information
        """
        healthy_nodes = sum(1 for node in self.worker_nodes.values() 
                           if node.status == NodeStatus.HEALTHY)
        
        total_capacity = sum(node.max_concurrent_tasks for node in self.worker_nodes.values()
                            if node.status == NodeStatus.HEALTHY)
        
        current_load = sum(node.current_tasks for node in self.worker_nodes.values()
                          if node.status == NodeStatus.HEALTHY)
        
        queue_length = len(self.active_tasks)
        
        return {
            "node_id": self.node_id,
            "is_coordinator": self.is_coordinator,
            "total_nodes": len(self.worker_nodes),
            "healthy_nodes": healthy_nodes,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "utilization": (current_load / total_capacity * 100) if total_capacity > 0 else 0,
            "queue_length": queue_length,
            "completed_tasks": len(self.completed_tasks),
            "scaling_strategy": self.scaling_strategy.value,
            "timestamp": time.time()
        }
    
    def trigger_scaling_decision(self):
        """Manually trigger scaling decision."""
        if not self.is_coordinator:
            logger.warning("Only coordinator can trigger scaling decisions")
            return
        
        metrics = self._collect_scaling_metrics()
        scaling_action = self._evaluate_scaling_decision(metrics)
        
        if scaling_action:
            self._execute_scaling_action(scaling_action)
    
    def _register_node(self):
        """Register this node in the distributed system."""
        node_info = {
            "node_id": self.node_id,
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
            "port": self.config.get("node_port", 8080),
            "capabilities": self.config.get("node_capabilities", ["inference"]),
            "max_concurrent_tasks": self.config.get("max_concurrent_tasks", mp.cpu_count()),
            "cpu_cores": mp.cpu_count(),
            "memory_gb": self._get_memory_info(),
            "gpu_count": self._get_gpu_count(),
            "status": NodeStatus.HEALTHY.value,
            "last_heartbeat": time.time(),
            "registered_at": time.time()
        }
        
        self.redis_client.set(f"node:{self.node_id}", json.dumps(node_info), ex=300)  # 5 minute TTL
        logger.info(f"Node registered: {self.node_id}")
    
    def _unregister_node(self):
        """Unregister this node from the distributed system."""
        self.redis_client.delete(f"node:{self.node_id}")
        logger.info(f"Node unregistered: {self.node_id}")
    
    def _get_memory_info(self) -> float:
        """Get system memory information in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8.0  # Default assumption
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n'))
        except:
            pass
        return 0
    
    def _start_heartbeat_thread(self):
        """Start heartbeat thread."""
        def heartbeat_loop():
            while not self._shutdown_event.is_set():
                try:
                    # Update heartbeat
                    self.redis_client.set(f"node:{self.node_id}:heartbeat", 
                                        time.time(), ex=self.heartbeat_interval * 3)
                    
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        self._threads.append(thread)
    
    def _start_health_monitor_thread(self):
        """Start health monitoring thread."""
        def health_monitor_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._update_node_statuses()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health monitor error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=health_monitor_loop, daemon=True)
        thread.start()
        self._threads.append(thread)
    
    def _start_task_processor_thread(self):
        """Start task processor thread."""
        def task_processor_loop():
            while not self._shutdown_event.is_set():
                try:
                    # Check for tasks in Redis queue
                    task_data = self.redis_client.brpop(["inference_tasks"], timeout=1)
                    if task_data:
                        queue_name, task_json = task_data
                        task_info = json.loads(task_json)
                        
                        # Process the task
                        self._process_inference_task(task_info)
                    
                except Exception as e:
                    logger.error(f"Task processor error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=task_processor_loop, daemon=True)
        thread.start()
        self._threads.append(thread)
    
    def _start_coordinator_threads(self):
        """Start coordinator-specific threads."""
        # Auto-scaling thread
        def scaling_loop():
            while not self._shutdown_event.is_set():
                try:
                    metrics = self._collect_scaling_metrics()
                    self.metrics_history.append(metrics)
                    
                    scaling_action = self._evaluate_scaling_decision(metrics)
                    if scaling_action:
                        self._execute_scaling_action(scaling_action)
                    
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Scaling loop error: {e}")
                    time.sleep(30)
        
        # Load balancing thread
        def load_balancing_loop():
            while not self._shutdown_event.is_set():
                try:
                    self.load_balancer.rebalance_load()
                    time.sleep(30)  # Rebalance every 30 seconds
                except Exception as e:
                    logger.error(f"Load balancing error: {e}")
                    time.sleep(30)
        
        scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        balancing_thread = threading.Thread(target=load_balancing_loop, daemon=True)
        
        scaling_thread.start()
        balancing_thread.start()
        
        self._threads.extend([scaling_thread, balancing_thread])
    
    def _start_async_loop(self):
        """Start async event loop."""
        def run_async_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            
            # Start async tasks
            self._async_tasks.append(
                self._async_loop.create_task(self._async_metrics_collection())
            )
            self._async_tasks.append(
                self._async_loop.create_task(self._async_performance_prediction())
            )
            
            try:
                self._async_loop.run_forever()
            except Exception as e:
                logger.error(f"Async loop error: {e}")
        
        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()
        self._threads.append(async_thread)
    
    def _update_node_statuses(self):
        """Update status of all worker nodes."""
        current_time = time.time()
        
        # Get all registered nodes
        for node_id in list(self.worker_nodes.keys()):
            # Check heartbeat
            heartbeat = self.redis_client.get(f"node:{node_id}:heartbeat")
            
            if heartbeat:
                last_heartbeat = float(heartbeat)
                
                if current_time - last_heartbeat > self.heartbeat_interval * 3:
                    # Node is unhealthy
                    self.worker_nodes[node_id].status = NodeStatus.UNHEALTHY
                    self.worker_nodes[node_id].last_heartbeat = last_heartbeat
                elif current_time - last_heartbeat > self.heartbeat_interval * 2:
                    # Node is degraded
                    self.worker_nodes[node_id].status = NodeStatus.DEGRADED
                    self.worker_nodes[node_id].last_heartbeat = last_heartbeat
                else:
                    # Node is healthy
                    self.worker_nodes[node_id].status = NodeStatus.HEALTHY
                    self.worker_nodes[node_id].last_heartbeat = last_heartbeat
            else:
                # No heartbeat found - node is offline
                if node_id in self.worker_nodes:
                    self.worker_nodes[node_id].status = NodeStatus.OFFLINE
        
        # Discover new nodes
        self._discover_new_nodes()
    
    def _discover_new_nodes(self):
        """Discover new worker nodes."""
        # This would scan Redis for new node registrations
        # For now, implement basic discovery
        pass
    
    def _process_inference_task(self, task_info: Dict[str, Any]):
        """Process an inference task.
        
        Args:
            task_info: Task information from queue
        """
        task_id = task_info["task_id"]
        
        try:
            # Deserialize input data
            input_data = self._deserialize_input_data(task_info["input_data"])
            
            # Simulate inference processing
            start_time = time.time()
            result = self._run_inference(
                task_info["model_name"], 
                task_info["task_type"], 
                input_data
            )
            
            processing_time = time.time() - start_time
            
            # Store result
            result_data = {
                "task_id": task_id,
                "result": self._serialize_result(result),
                "processing_time": processing_time,
                "completed_at": time.time(),
                "node_id": self.node_id
            }
            
            self.redis_client.set(f"task_result:{task_id}", 
                                json.dumps(result_data), ex=3600)  # 1 hour TTL
            
            # Update local tracking
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.result = result
                task.completed_at = time.time()
                self.completed_tasks.append(task)
                del self.active_tasks[task_id]
            
            logger.debug(f"Task completed: {task_id} in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Task processing failed: {task_id}: {e}")
            
            # Store error
            error_data = {
                "task_id": task_id,
                "error": str(e),
                "completed_at": time.time(),
                "node_id": self.node_id
            }
            
            self.redis_client.set(f"task_result:{task_id}", 
                                json.dumps(error_data), ex=3600)
            
            # Update local tracking
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.error = str(e)
                task.completed_at = time.time()
                self.completed_tasks.append(task)
                del self.active_tasks[task_id]
    
    def _run_inference(self, model_name: str, task_type: str, input_data: Any) -> Any:
        """Run model inference.
        
        Args:
            model_name: Name of the model
            task_type: Type of task
            input_data: Input data
            
        Returns:
            Inference result
        """
        # This would integrate with actual model inference
        # For now, simulate inference
        import random
        processing_time = random.uniform(0.1, 2.0)  # Simulate variable processing time
        time.sleep(processing_time)
        
        return {
            "task_type": task_type,
            "model_name": model_name,
            "result": f"Mock {task_type} result for {model_name}",
            "confidence": random.uniform(0.7, 0.95),
            "processing_time": processing_time
        }
    
    def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect metrics for scaling decisions."""
        current_time = time.time()
        
        # Count healthy nodes
        healthy_nodes = sum(1 for node in self.worker_nodes.values() 
                           if node.status == NodeStatus.HEALTHY)
        
        # Calculate queue length
        queue_length = len(self.active_tasks)
        
        # Calculate average response time from recent tasks
        recent_tasks = [task for task in self.completed_tasks 
                       if current_time - task.completed_at < 300]  # Last 5 minutes
        
        if recent_tasks:
            response_times = [(task.completed_at - task.created_at) 
                            for task in recent_tasks if task.completed_at and task.created_at]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        else:
            avg_response_time = 0
        
        # Estimate system utilization
        total_capacity = sum(node.max_concurrent_tasks for node in self.worker_nodes.values()
                            if node.status == NodeStatus.HEALTHY)
        current_load = sum(node.current_tasks for node in self.worker_nodes.values()
                          if node.status == NodeStatus.HEALTHY)
        
        cpu_utilization = (current_load / total_capacity * 100) if total_capacity > 0 else 0
        
        # Calculate requests per second
        recent_completed = len([task for task in self.completed_tasks 
                              if current_time - task.completed_at < 60])  # Last minute
        requests_per_second = recent_completed / 60.0
        
        # Calculate error rate
        recent_errors = len([task for task in self.completed_tasks 
                           if task.error and current_time - task.completed_at < 300])
        error_rate = (recent_errors / len(recent_tasks)) if recent_tasks else 0
        
        metrics = ScalingMetrics(
            timestamp=current_time,
            total_nodes=len(self.worker_nodes),
            healthy_nodes=healthy_nodes,
            queue_length=queue_length,
            average_response_time=avg_response_time,
            cpu_utilization=cpu_utilization,
            memory_utilization=0,  # Would implement actual memory monitoring
            requests_per_second=requests_per_second,
            error_rate=error_rate
        )
        
        return metrics
    
    def _evaluate_scaling_decision(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """Evaluate whether scaling action is needed.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Scaling action or None if no action needed
        """
        # Reactive scaling logic
        if self.scaling_strategy in [ScalingStrategy.REACTIVE, ScalingStrategy.HYBRID]:
            # Scale up conditions
            if (metrics.cpu_utilization > self.target_cpu_utilization * self.scale_up_threshold or
                metrics.queue_length > self.target_queue_length * self.scale_up_threshold or
                metrics.average_response_time > 5.0):  # 5 second response time threshold
                
                if metrics.healthy_nodes < self.max_nodes:
                    return {
                        "action": "scale_up",
                        "reason": f"High utilization: CPU={metrics.cpu_utilization:.1f}%, Queue={metrics.queue_length}",
                        "target_nodes": min(metrics.healthy_nodes + 1, self.max_nodes)
                    }
            
            # Scale down conditions
            elif (metrics.cpu_utilization < self.target_cpu_utilization * self.scale_down_threshold and
                  metrics.queue_length < self.target_queue_length * self.scale_down_threshold and
                  metrics.average_response_time < 1.0):  # Fast response time
                
                if metrics.healthy_nodes > self.min_nodes:
                    return {
                        "action": "scale_down",
                        "reason": f"Low utilization: CPU={metrics.cpu_utilization:.1f}%, Queue={metrics.queue_length}",
                        "target_nodes": max(metrics.healthy_nodes - 1, self.min_nodes)
                    }
        
        # Predictive scaling (if enabled)
        if self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            predicted_load = self._predict_future_load(metrics)
            if predicted_load and predicted_load > metrics.cpu_utilization * 1.5:
                return {
                    "action": "scale_up",
                    "reason": f"Predicted load increase: {predicted_load:.1f}%",
                    "target_nodes": min(metrics.healthy_nodes + 1, self.max_nodes)
                }
        
        return None
    
    def _predict_future_load(self, current_metrics: ScalingMetrics) -> Optional[float]:
        """Predict future load based on historical data.
        
        Args:
            current_metrics: Current system metrics
            
        Returns:
            Predicted CPU utilization percentage
        """
        if len(self.metrics_history) < 10:
            return None
        
        # Simple linear trend prediction
        recent_metrics = list(self.metrics_history)[-10:]
        timestamps = [m.timestamp for m in recent_metrics]
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        
        # Calculate trend
        if len(timestamps) >= 2:
            time_diff = timestamps[-1] - timestamps[0]
            cpu_diff = cpu_values[-1] - cpu_values[0]
            
            if time_diff > 0:
                trend = cpu_diff / time_diff  # CPU change per second
                
                # Predict 5 minutes ahead
                prediction_window = 300  # 5 minutes
                predicted_cpu = cpu_values[-1] + (trend * prediction_window)
                
                return max(0, min(100, predicted_cpu))
        
        return None
    
    def _execute_scaling_action(self, action: Dict[str, Any]):
        """Execute scaling action.
        
        Args:
            action: Scaling action to execute
        """
        action_type = action["action"]
        target_nodes = action["target_nodes"]
        reason = action["reason"]
        
        logger.info(f"Executing scaling action: {action_type} to {target_nodes} nodes. Reason: {reason}")
        
        if action_type == "scale_up":
            self._scale_up_nodes(target_nodes)
        elif action_type == "scale_down":
            self._scale_down_nodes(target_nodes)
        
        # Publish scaling event
        scaling_event = {
            "timestamp": time.time(),
            "action": action_type,
            "target_nodes": target_nodes,
            "reason": reason,
            "coordinator": self.node_id
        }
        
        self.redis_client.publish("scaling_events", json.dumps(scaling_event))
    
    def _scale_up_nodes(self, target_count: int):
        """Scale up to target node count.
        
        Args:
            target_count: Target number of nodes
        """
        current_healthy = sum(1 for node in self.worker_nodes.values() 
                            if node.status == NodeStatus.HEALTHY)
        
        nodes_to_add = target_count - current_healthy
        
        for i in range(nodes_to_add):
            # In a real implementation, this would:
            # 1. Start new container/VM instances
            # 2. Deploy application to new instances
            # 3. Register new nodes
            
            # For simulation, create mock node
            new_node_id = str(uuid.uuid4())
            new_node = WorkerNode(
                node_id=new_node_id,
                hostname=f"worker-{len(self.worker_nodes) + 1}",
                port=8080,
                capabilities=["inference"],
                max_concurrent_tasks=mp.cpu_count(),
                cpu_cores=mp.cpu_count(),
                memory_gb=8.0,
                status=NodeStatus.HEALTHY,
                last_heartbeat=time.time()
            )
            
            self.worker_nodes[new_node_id] = new_node
            
            # Register in Redis
            node_info = asdict(new_node)
            self.redis_client.set(f"node:{new_node_id}", json.dumps(node_info, default=str), ex=300)
            
            logger.info(f"Scaled up: Added node {new_node_id}")
    
    def _scale_down_nodes(self, target_count: int):
        """Scale down to target node count.
        
        Args:
            target_count: Target number of nodes
        """
        current_healthy = sum(1 for node in self.worker_nodes.values() 
                            if node.status == NodeStatus.HEALTHY)
        
        nodes_to_remove = current_healthy - target_count
        
        # Select nodes with least load for removal
        healthy_nodes = [(node_id, node) for node_id, node in self.worker_nodes.items() 
                        if node.status == NodeStatus.HEALTHY]
        
        # Sort by current tasks (ascending) to remove least loaded nodes
        healthy_nodes.sort(key=lambda x: x[1].current_tasks)
        
        for i in range(min(nodes_to_remove, len(healthy_nodes))):
            node_id, node = healthy_nodes[i]
            
            # In a real implementation, this would:
            # 1. Drain tasks from the node
            # 2. Gracefully shutdown the node
            # 3. Terminate container/VM instance
            
            # For simulation, mark as offline
            node.status = NodeStatus.OFFLINE
            
            # Remove from Redis
            self.redis_client.delete(f"node:{node_id}")
            
            logger.info(f"Scaled down: Removed node {node_id}")
    
    async def _async_metrics_collection(self):
        """Async metrics collection task."""
        while not self._shutdown_event.is_set():
            try:
                # Collect detailed metrics asynchronously
                await self._collect_detailed_metrics()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Async metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _async_performance_prediction(self):
        """Async performance prediction task."""
        while not self._shutdown_event.is_set():
            try:
                # Update performance prediction models
                await self._update_performance_models()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Async performance prediction error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_detailed_metrics(self):
        """Collect detailed system metrics asynchronously."""
        # This would implement detailed metrics collection
        # including network I/O, disk usage, etc.
        pass
    
    async def _update_performance_models(self):
        """Update performance prediction models."""
        # This would implement machine learning models for
        # performance prediction and optimization
        pass
    
    def _serialize_input_data(self, data: Any) -> str:
        """Serialize input data for distributed queue."""
        if isinstance(data, (dict, list, str, int, float, bool)):
            return json.dumps(data)
        elif hasattr(data, '__dict__'):
            return json.dumps(data.__dict__)
        else:
            return str(data)
    
    def _deserialize_input_data(self, data_str: str) -> Any:
        """Deserialize input data from distributed queue."""
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return data_str
    
    def _serialize_result(self, result: Any) -> str:
        """Serialize result for storage."""
        return json.dumps(result, default=str)
    
    def _deserialize_result(self, result_str: str) -> Any:
        """Deserialize result from storage."""
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            return result_str


class LoadBalancer:
    """Intelligent load balancer for distributed inference."""
    
    def __init__(self, scaling_system: DistributedScaling):
        """Initialize load balancer.
        
        Args:
            scaling_system: Parent scaling system
        """
        self.scaling_system = scaling_system
        self.balancing_algorithms = {
            "round_robin": self._round_robin_balance,
            "least_loaded": self._least_loaded_balance,
            "weighted_random": self._weighted_random_balance,
            "capability_based": self._capability_based_balance
        }
        
        self.current_algorithm = "least_loaded"
        self.round_robin_counter = 0
        
    def select_node_for_task(self, task: InferenceTask) -> Optional[str]:
        """Select optimal node for task execution.
        
        Args:
            task: Inference task to assign
            
        Returns:
            Selected node ID or None if no suitable node
        """
        # Get healthy nodes capable of handling this task
        suitable_nodes = []
        
        for node_id, node in self.scaling_system.worker_nodes.items():
            if (node.status == NodeStatus.HEALTHY and
                task.task_type in node.capabilities and
                node.current_tasks < node.max_concurrent_tasks):
                suitable_nodes.append((node_id, node))
        
        if not suitable_nodes:
            return None
        
        # Apply load balancing algorithm
        algorithm = self.balancing_algorithms.get(self.current_algorithm)
        if algorithm:
            return algorithm(suitable_nodes, task)
        
        # Fallback to first available node
        return suitable_nodes[0][0]
    
    def _round_robin_balance(self, nodes: List[Tuple[str, WorkerNode]], 
                           task: InferenceTask) -> str:
        """Round-robin load balancing."""
        if not nodes:
            return None
        
        selected = nodes[self.round_robin_counter % len(nodes)]
        self.round_robin_counter += 1
        return selected[0]
    
    def _least_loaded_balance(self, nodes: List[Tuple[str, WorkerNode]], 
                            task: InferenceTask) -> str:
        """Least loaded node balancing."""
        if not nodes:
            return None
        
        # Sort by current load (ascending)
        nodes.sort(key=lambda x: x[1].current_tasks / x[1].max_concurrent_tasks)
        return nodes[0][0]
    
    def _weighted_random_balance(self, nodes: List[Tuple[str, WorkerNode]], 
                               task: InferenceTask) -> str:
        """Weighted random selection based on available capacity."""
        if not nodes:
            return None
        
        # Calculate weights based on available capacity
        weights = []
        for node_id, node in nodes:
            available_capacity = node.max_concurrent_tasks - node.current_tasks
            weight = available_capacity / node.max_concurrent_tasks
            weights.append(weight)
        
        # Weighted random selection
        import random
        total_weight = sum(weights)
        if total_weight == 0:
            return nodes[0][0]
        
        random_value = random.random() * total_weight
        cumulative_weight = 0
        
        for i, (node_id, node) in enumerate(nodes):
            cumulative_weight += weights[i]
            if random_value <= cumulative_weight:
                return node_id
        
        return nodes[-1][0]
    
    def _capability_based_balance(self, nodes: List[Tuple[str, WorkerNode]], 
                                task: InferenceTask) -> str:
        """Capability-based balancing with specialization."""
        if not nodes:
            return None
        
        # Prefer nodes with specialized capabilities for the task
        specialized_nodes = []
        general_nodes = []
        
        for node_id, node in nodes:
            if f"{task.task_type}_specialized" in node.capabilities:
                specialized_nodes.append((node_id, node))
            else:
                general_nodes.append((node_id, node))
        
        # Use specialized nodes first, then general nodes
        candidate_nodes = specialized_nodes if specialized_nodes else general_nodes
        
        # Among candidates, select least loaded
        if candidate_nodes:
            candidate_nodes.sort(key=lambda x: x[1].current_tasks / x[1].max_concurrent_tasks)
            return candidate_nodes[0][0]
        
        return nodes[0][0]
    
    def rebalance_load(self):
        """Rebalance load across nodes."""
        # Check for severely imbalanced nodes
        healthy_nodes = [(node_id, node) for node_id, node in self.scaling_system.worker_nodes.items()
                        if node.status == NodeStatus.HEALTHY]
        
        if len(healthy_nodes) < 2:
            return
        
        # Calculate load distribution
        load_ratios = [(node_id, node.current_tasks / node.max_concurrent_tasks) 
                      for node_id, node in healthy_nodes]
        
        load_ratios.sort(key=lambda x: x[1])
        
        # Check if rebalancing is needed
        min_load = load_ratios[0][1]
        max_load = load_ratios[-1][1]
        
        if max_load - min_load > 0.3:  # 30% load difference threshold
            logger.info(f"Load imbalance detected: {min_load:.2f} to {max_load:.2f}")
            # In a real implementation, this would trigger task migration
    
    def update_balancing_algorithm(self, algorithm: str):
        """Update load balancing algorithm.
        
        Args:
            algorithm: New balancing algorithm name
        """
        if algorithm in self.balancing_algorithms:
            self.current_algorithm = algorithm
            logger.info(f"Load balancing algorithm updated to: {algorithm}")
        else:
            logger.warning(f"Unknown load balancing algorithm: {algorithm}")


class GlobalDeploymentOrchestrator:
    """Global deployment orchestration for multi-region scaling."""
    
    def __init__(self, distributed_scaling: DistributedScaling):
        """Initialize global deployment orchestrator.
        
        Args:
            distributed_scaling: Parent distributed scaling system
        """
        self.distributed_scaling = distributed_scaling
        self.regions = {}
        self.region_coordinators = {}
        self.traffic_routing = TrafficRouter()
        
    def register_region(self, region_name: str, coordinator_endpoint: str, 
                       capabilities: List[str] = None):
        """Register a new deployment region.
        
        Args:
            region_name: Name of the region
            coordinator_endpoint: Coordinator endpoint for the region
            capabilities: Regional capabilities
        """
        self.regions[region_name] = {
            "coordinator_endpoint": coordinator_endpoint,
            "capabilities": capabilities or [],
            "status": "active",
            "last_heartbeat": time.time(),
            "registered_at": time.time()
        }
        
        logger.info(f"Region registered: {region_name}")
    
    def route_global_request(self, request: InferenceTask, 
                           source_region: str = None) -> str:
        """Route request to optimal region.
        
        Args:
            request: Inference request
            source_region: Source region of the request
            
        Returns:
            Selected region name
        """
        return self.traffic_routing.select_optimal_region(
            request, source_region, self.regions
        )
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global deployment status.
        
        Returns:
            Global status information
        """
        return {
            "total_regions": len(self.regions),
            "active_regions": sum(1 for r in self.regions.values() if r["status"] == "active"),
            "regions": list(self.regions.keys()),
            "timestamp": time.time()
        }


class TrafficRouter:
    """Intelligent traffic routing for global deployments."""
    
    def select_optimal_region(self, request: InferenceTask, 
                            source_region: Optional[str],
                            regions: Dict[str, Any]) -> str:
        """Select optimal region for request processing.
        
        Args:
            request: Inference request
            source_region: Source region
            regions: Available regions
            
        Returns:
            Selected region name
        """
        if not regions:
            raise ValueError("No regions available")
        
        # Simple routing: prefer source region if available
        if source_region and source_region in regions:
            region_info = regions[source_region]
            if (region_info["status"] == "active" and 
                request.task_type in region_info.get("capabilities", [])):
                return source_region
        
        # Fallback to first available region
        for region_name, region_info in regions.items():
            if (region_info["status"] == "active" and 
                request.task_type in region_info.get("capabilities", [])):
                return region_name
        
        # Last resort: any active region
        active_regions = [name for name, info in regions.items() if info["status"] == "active"]
        if active_regions:
            return active_regions[0]
        
        raise ValueError("No active regions available")


def create_default_scaling_config() -> Dict[str, Any]:
    """Create default scaling configuration."""
    return {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "scaling_strategy": "hybrid",
        "min_nodes": 1,
        "max_nodes": 10,
        "target_cpu_utilization": 70.0,
        "target_queue_length": 10,
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.3,
        "health_check_interval": 30,
        "heartbeat_interval": 10,
        "node_port": 8080,
        "node_capabilities": ["inference", "caption", "ocr", "vqa"],
        "max_concurrent_tasks": mp.cpu_count(),
        "load_balancing_algorithm": "least_loaded"
    }