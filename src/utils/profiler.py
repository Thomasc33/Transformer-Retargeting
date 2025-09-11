"""
Comprehensive profiling utilities for training optimization.
"""

import time
import torch
import psutil
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional
import logging

class PerformanceProfiler:
    """
    Comprehensive performance profiler for training optimization.
    Tracks GPU utilization, memory usage, and timing for each component.
    """
    
    def __init__(self, log_interval: int = 100, max_history: int = 1000):
        self.log_interval = log_interval
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # Timing data
        self.timers = defaultdict(list)
        self.current_timers = {}
        
        # Memory tracking
        self.memory_history = deque(maxlen=max_history)
        self.gpu_memory_history = deque(maxlen=max_history)
        
        # GPU utilization tracking
        self.gpu_util_history = deque(maxlen=max_history)
        
        # Batch processing stats
        self.batch_times = deque(maxlen=max_history)
        self.batch_sizes = deque(maxlen=max_history)
        
        # Component-specific timing
        self.component_times = defaultdict(lambda: deque(maxlen=max_history))
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
    def start_timer(self, name: str):
        """Start timing a component."""
        self.current_timers[name] = time.time()
        
    def end_timer(self, name: str):
        """End timing a component and record the duration."""
        if name in self.current_timers:
            duration = time.time() - self.current_timers[name]
            self.timers[name].append(duration)
            self.component_times[name].append(duration)
            del self.current_timers[name]
            return duration
        return 0.0
        
    def record_batch_stats(self, batch_size: int, batch_time: float):
        """Record batch processing statistics."""
        self.batch_times.append(batch_time)
        self.batch_sizes.append(batch_size)
        
    def record_memory_usage(self):
        """Record current memory usage."""
        # CPU memory
        cpu_memory = psutil.virtual_memory().percent
        self.memory_history.append(cpu_memory)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            self.gpu_memory_history.append(gpu_memory)
            
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring of system resources."""
        if self.monitoring:
            return
            
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                self.record_memory_usage()
                time.sleep(interval)
                
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def get_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        stats = {
            'timing': {},
            'memory': {},
            'batch_processing': {},
            'gpu': {}
        }
        
        # Timing statistics
        for name, times in self.timers.items():
            if times:
                stats['timing'][name] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'count': len(times)
                }
                
        # Memory statistics
        if self.memory_history:
            stats['memory']['cpu_avg'] = sum(self.memory_history) / len(self.memory_history)
            stats['memory']['cpu_max'] = max(self.memory_history)
            
        if self.gpu_memory_history:
            stats['memory']['gpu_avg'] = sum(self.gpu_memory_history) / len(self.gpu_memory_history)
            stats['memory']['gpu_max'] = max(self.gpu_memory_history)
            
        # Batch processing statistics
        if self.batch_times:
            stats['batch_processing']['avg_time'] = sum(self.batch_times) / len(self.batch_times)
            stats['batch_processing']['min_time'] = min(self.batch_times)
            stats['batch_processing']['max_time'] = max(self.batch_times)
            
        if self.batch_sizes:
            stats['batch_processing']['avg_size'] = sum(self.batch_sizes) / len(self.batch_sizes)
            
        # GPU statistics
        if torch.cuda.is_available():
            stats['gpu']['device_count'] = torch.cuda.device_count()
            stats['gpu']['current_device'] = torch.cuda.current_device()
            stats['gpu']['memory_allocated'] = torch.cuda.memory_allocated()
            stats['gpu']['memory_reserved'] = torch.cuda.memory_reserved()
            stats['gpu']['max_memory_allocated'] = torch.cuda.max_memory_allocated()
            
        return stats
        
    def print_summary(self):
        """Print a comprehensive performance summary."""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*80)
        
        # Timing summary
        if stats['timing']:
            print("\n📊 TIMING ANALYSIS:")
            for name, timing in stats['timing'].items():
                print(f"  {name:30s}: {timing['avg']:8.4f}s avg ({timing['count']:4d} calls)")
                
        # Memory summary
        if stats['memory']:
            print("\n💾 MEMORY USAGE:")
            if 'cpu_avg' in stats['memory']:
                print(f"  CPU Memory Average: {stats['memory']['cpu_avg']:6.2f}%")
                print(f"  CPU Memory Peak:    {stats['memory']['cpu_max']:6.2f}%")
            if 'gpu_avg' in stats['memory']:
                print(f"  GPU Memory Average: {stats['memory']['gpu_avg']:6.2f}%")
                print(f"  GPU Memory Peak:    {stats['memory']['gpu_max']:6.2f}%")
                
        # Batch processing summary
        if stats['batch_processing']:
            print("\n⚡ BATCH PROCESSING:")
            if 'avg_time' in stats['batch_processing']:
                print(f"  Average Batch Time: {stats['batch_processing']['avg_time']:8.4f}s")
                print(f"  Min Batch Time:     {stats['batch_processing']['min_time']:8.4f}s")
                print(f"  Max Batch Time:     {stats['batch_processing']['max_time']:8.4f}s")
            if 'avg_size' in stats['batch_processing']:
                print(f"  Average Batch Size: {stats['batch_processing']['avg_size']:8.2f}")
                
        # GPU summary
        if stats['gpu']:
            print("\n🖥️  GPU INFORMATION:")
            print(f"  Device Count:       {stats['gpu']['device_count']}")
            print(f"  Current Device:     {stats['gpu']['current_device']}")
            print(f"  Memory Allocated:   {stats['gpu']['memory_allocated'] / 1024**3:.2f} GB")
            print(f"  Memory Reserved:    {stats['gpu']['memory_reserved'] / 1024**3:.2f} GB")
            print(f"  Max Memory Used:    {stats['gpu']['max_memory_allocated'] / 1024**3:.2f} GB")
            
        print("="*80)
        
    def get_bottlenecks(self, top_n: int = 5) -> List[tuple]:
        """Identify the top bottlenecks in the system."""
        bottlenecks = []
        
        for name, times in self.timers.items():
            if times:
                total_time = sum(times)
                avg_time = total_time / len(times)
                bottlenecks.append((name, total_time, avg_time, len(times)))
                
        # Sort by total time (biggest bottlenecks first)
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        return bottlenecks[:top_n]
        
    def reset(self):
        """Reset all profiling data."""
        self.timers.clear()
        self.current_timers.clear()
        self.memory_history.clear()
        self.gpu_memory_history.clear()
        self.gpu_util_history.clear()
        self.batch_times.clear()
        self.batch_sizes.clear()
        self.component_times.clear()


# Global profiler instance
global_profiler = PerformanceProfiler()


def profile_function(name: str):
    """Decorator to profile function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global_profiler.start_timer(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                global_profiler.end_timer(name)
        return wrapper
    return decorator


class ProfilerContext:
    """Context manager for profiling code blocks."""
    
    def __init__(self, name: str, profiler: Optional[PerformanceProfiler] = None):
        self.name = name
        self.profiler = profiler or global_profiler
        
    def __enter__(self):
        self.profiler.start_timer(self.name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.name)
