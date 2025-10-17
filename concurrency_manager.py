import threading
import time
import uuid
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    task_id: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    function: Optional[Callable] = None
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

class ConcurrencyManager:
    def __init__(self, max_concurrent: int = 2):
        """
        并发控制管理器
        
        Args:
            max_concurrent: 最大并发数量
        """
        self.max_concurrent = max_concurrent
        self.running_tasks: Dict[str, Task] = {}
        self.queued_tasks: List[Task] = []
        self.completed_tasks: Dict[str, Task] = {}
        self.lock = threading.RLock()
        self.worker_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        
        # 启动工作线程
        self._start_workers()
    
    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_concurrent):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
    
    def _worker_loop(self):
        """工作线程主循环"""
        while not self.shutdown_event.is_set():
            try:
                task = self._get_next_task()
                if task:
                    self._execute_task(task)
                else:
                    # 没有任务时短暂休眠
                    time.sleep(0.1)
            except Exception as e:
                print(f"Worker thread error: {e}")
                time.sleep(1)
    
    def _get_next_task(self) -> Optional[Task]:
        """获取下一个要执行的任务"""
        with self.lock:
            if self.queued_tasks:
                return self.queued_tasks.pop(0)
            return None
    
    def _execute_task(self, task: Task):
        """执行任务"""
        try:
            with self.lock:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                self.running_tasks[task.task_id] = task
            
            # 执行任务
            if task.function:
                result = task.function(*task.args, **task.kwargs)
                task.result = result
            
            # 标记完成
            with self.lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                self.completed_tasks[task.task_id] = task
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
            print(f"Task {task.task_id} completed")
        except Exception as e:
            # 标记失败
            print(f"Task {task.task_id} failed: {e}")
            with self.lock:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.error = str(e)
                self.completed_tasks[task.task_id] = task
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """
        提交任务
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            task_id: 任务ID
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            created_at=time.time(),
            function=func,
            args=args,
            kwargs=kwargs
        )
        
        with self.lock:
            self.queued_tasks.append(task)
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """获取任务状态"""
        with self.lock:
            if task_id in self.running_tasks:
                return self.running_tasks[task_id]
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            else:
                # 检查队列中的任务
                for task in self.queued_tasks:
                    if task.task_id == task_id:
                        return task
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        with self.lock:
            return {
                "max_concurrent": self.max_concurrent,
                "running_count": len(self.running_tasks),
                "queued_count": len(self.queued_tasks),
                "completed_count": len(self.completed_tasks),
                "running_tasks": [task.task_id for task in self.running_tasks.values()],
                "queued_tasks": [task.task_id for task in self.queued_tasks],
            }
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Task:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            Task: 完成的任务
        """
        start_time = time.time()
        
        while True:
            task = self.get_task_status(task_id)
            if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return task
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
            
            time.sleep(0.1)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧任务"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self.lock:
            # 清理已完成的任务
            old_tasks = [
                task_id for task_id, task in self.completed_tasks.items()
                if current_time - task.completed_at > max_age_seconds
            ]
            for task_id in old_tasks:
                del self.completed_tasks[task_id]
    
    def shutdown(self):
        """关闭管理器"""
        self.shutdown_event.set()
        for worker in self.worker_threads:
            worker.join(timeout=5)
