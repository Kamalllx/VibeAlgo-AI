
# backend/utils/performance_monitor.py
import time
import functools
import psutil
from typing import Dict, Any, Callable,List
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.start_time = time.time()
    
    def monitor_function(self, func: Callable) -> Callable:
        """Decorator to monitor function performance"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                metric = {
                    'function_name': func.__name__,
                    'execution_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'timestamp': datetime.now().isoformat(),
                    'success': success,
                    'error': error
                }
                
                self.metrics.append(metric)
                
                # Keep only last 100 metrics
                if len(self.metrics) > 100:
                    self.metrics = self.metrics[-100:]
            
            return result
        return wrapper
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.metrics:
            return {'total_calls': 0, 'average_time': 0, 'total_time': 0}
        
        successful_metrics = [m for m in self.metrics if m['success']]
        
        if not successful_metrics:
            return {'total_calls': len(self.metrics), 'success_rate': 0}
        
        total_time = sum(m['execution_time'] for m in successful_metrics)
        avg_time = total_time / len(successful_metrics)
        
        return {
            'total_calls': len(self.metrics),
            'successful_calls': len(successful_metrics),
            'success_rate': len(successful_metrics) / len(self.metrics) * 100,
            'average_execution_time': avg_time,
            'total_execution_time': total_time,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance metrics"""
        return self.metrics[-count:]

# Global monitor instance
monitor = PerformanceMonitor()

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    return monitor.monitor_function(func)
