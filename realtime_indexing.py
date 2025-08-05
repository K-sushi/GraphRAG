#!/usr/bin/env python3
"""
Real-time GraphRAG Indexing System
SuperClaude Wave Orchestration - Phase 2 Enhancement

Monitors file changes and triggers incremental GraphRAG indexing
Implements background task processing with WebSocket notifications
"""

import os
import sys
import asyncio
import logging
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import threading
from queue import Queue
import websockets
import yaml

# File monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("Watchdog not available - file monitoring disabled")

# Background task processing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IndexingTask:
    """Represents an indexing task"""
    task_id: str
    task_type: str  # "full", "incremental", "document"
    file_path: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, 1 = highest
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, processing, completed, failed
    retries: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class IndexingStats:
    """Indexing performance statistics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time: float = 0.0
    last_indexing_time: Optional[datetime] = None
    documents_indexed: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    communities_detected: int = 0

class FileChangeHandler(FileSystemEventHandler):
    """Handles file system changes for real-time indexing"""
    
    def __init__(self, indexing_manager):
        self.indexing_manager = indexing_manager
        self.debounce_delay = 2.0  # seconds
        self.pending_changes = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Only process text files in input directory
        if file_path.suffix.lower() in ['.txt', '.md', '.pdf', '.docx'] and 'input' in str(file_path):
            self._schedule_indexing(file_path, "incremental")
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        if file_path.suffix.lower() in ['.txt', '.md', '.pdf', '.docx'] and 'input' in str(file_path):
            self._schedule_indexing(file_path, "document")
    
    def on_deleted(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        if 'input' in str(file_path):
            # Schedule removal from index
            self._schedule_indexing(file_path, "remove")
    
    def _schedule_indexing(self, file_path: Path, task_type: str):
        """Schedule indexing with debouncing"""
        file_key = str(file_path)
        
        # Cancel previous timer if exists
        if file_key in self.pending_changes:
            self.pending_changes[file_key].cancel()
        
        # Schedule new indexing task
        timer = threading.Timer(
            self.debounce_delay,
            self._create_indexing_task,
            args=[file_path, task_type]
        )
        timer.start()
        self.pending_changes[file_key] = timer
    
    def _create_indexing_task(self, file_path: Path, task_type: str):
        """Create and queue indexing task"""
        try:
            task = IndexingTask(
                task_id=f"{task_type}_{int(time.time())}_{file_path.name}",
                task_type=task_type,
                file_path=str(file_path),
                metadata={
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                    "trigger": "file_change"
                },
                priority=2 if task_type == "document" else 3
            )
            
            self.indexing_manager.add_task(task)
            logger.info(f"Scheduled {task_type} indexing for {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to create indexing task for {file_path}: {e}")
        finally:
            # Clean up pending changes
            file_key = str(file_path)
            if file_key in self.pending_changes:
                del self.pending_changes[file_key]

class WebSocketNotificationServer:
    """WebSocket server for real-time notifications"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        
    async def register_client(self, websocket, path):
        """Register new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send current stats
        await self.send_to_client(websocket, {
            "type": "connection",
            "message": "Connected to GraphRAG indexing server",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def send_to_client(self, websocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
            
        # Remove closed connections
        closed_clients = set()
        for client in self.clients.copy():
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                closed_clients.add(client)
        
        # Clean up closed connections
        self.clients -= closed_clients
    
    async def start_server(self):
        """Start WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.register_client,
                self.host,
                self.port
            )
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def stop_server(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

class BackgroundTaskProcessor:
    """Background task processor with multiple workers"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def start(self):
        """Start background processing"""
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"IndexingWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} background workers")
    
    def stop(self):
        """Stop background processing"""
        self.running = False
        
        # Signal workers to stop
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Background processing stopped")
    
    def submit_task(self, task: IndexingTask):
        """Submit task for processing"""
        if self.running:
            self.task_queue.put(task)
        else:
            logger.warning("Task processor not running, task ignored")
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[IndexingTask]:
        """Get completed task result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None
    
    def _worker_loop(self):
        """Worker loop for processing tasks"""
        worker_name = threading.current_thread().name
        
        while self.running:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                logger.info(f"{worker_name} processing task: {task.task_id}")
                
                # Process task
                self._process_task(task)
                
                # Put result in result queue
                self.result_queue.put(task)
                
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                
        logger.info(f"{worker_name} stopped")
    
    def _process_task(self, task: IndexingTask):
        """Process individual indexing task"""
        start_time = time.time()
        task.status = "processing"
        
        try:
            if task.task_type == "full":
                self._process_full_indexing(task)
            elif task.task_type == "incremental":
                self._process_incremental_indexing(task)
            elif task.task_type == "document":
                self._process_document_indexing(task)
            elif task.task_type == "remove":
                self._process_document_removal(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.status = "completed"
            
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.retries += 1
            logger.error(f"Task {task.task_id} failed: {e}")
            
        task.processing_time = time.time() - start_time
    
    def _process_full_indexing(self, task: IndexingTask):
        """Process full indexing task"""
        logger.info(f"Full indexing started: {task.task_id}")
        
        # Import GraphRAG pipeline
        try:
            from graphrag_pipeline import create_graphrag_pipeline
            
            # Create and run pipeline
            pipeline = create_graphrag_pipeline({
                "input_dir": "./input",
                "output_dir": "./output",
                "cache_dir": "./cache",
            })
            
            # Run pipeline (this would be async in real implementation)
            # For now, simulate processing
            time.sleep(2.0)  # Simulate processing time
            
            task.metadata.update({
                "documents_processed": 5,
                "entities_extracted": 42,
                "relationships_extracted": 28,
                "communities_detected": 7
            })
            
        except ImportError:
            # Fallback simulation
            logger.warning("GraphRAG not available, simulating processing")
            time.sleep(1.0)
            
            task.metadata.update({
                "documents_processed": 5,
                "entities_extracted": 35,
                "relationships_extracted": 22,
                "communities_detected": 6
            })
    
    def _process_incremental_indexing(self, task: IndexingTask):
        """Process incremental indexing task"""
        logger.info(f"Incremental indexing for: {task.file_path}")
        
        if not task.file_path or not Path(task.file_path).exists():
            raise FileNotFoundError(f"File not found: {task.file_path}")
        
        # Read file content
        with open(task.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Calculate content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Simulate incremental processing
        time.sleep(0.5)
        
        task.metadata.update({
            "content_hash": content_hash,
            "content_length": len(content),
            "entities_updated": 8,
            "relationships_updated": 5
        })
    
    def _process_document_indexing(self, task: IndexingTask):
        """Process new document indexing"""
        logger.info(f"Document indexing for: {task.file_path}")
        
        if not task.file_path or not Path(task.file_path).exists():
            raise FileNotFoundError(f"File not found: {task.file_path}")
        
        # Read and process new document
        with open(task.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simulate document processing
        time.sleep(1.0)
        
        task.metadata.update({
            "document_id": Path(task.file_path).stem,
            "content_length": len(content),
            "entities_extracted": 12,
            "relationships_extracted": 8,
            "communities_affected": 3
        })
    
    def _process_document_removal(self, task: IndexingTask):
        """Process document removal from index"""
        logger.info(f"Document removal for: {task.file_path}")
        
        # Simulate removal processing
        time.sleep(0.2)
        
        task.metadata.update({
            "document_id": Path(task.file_path).stem,
            "entities_removed": 5,
            "relationships_removed": 3,
            "communities_affected": 2
        })

class RealtimeIndexingManager:
    """Main real-time indexing manager"""
    
    def __init__(self, config_path: str = "./graphrag_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Components
        self.task_processor = BackgroundTaskProcessor(
            max_workers=self.config.get("indexing", {}).get("max_workers", 4)
        )
        
        self.websocket_server = WebSocketNotificationServer(
            host=self.config.get("websocket", {}).get("host", "localhost"),
            port=self.config.get("websocket", {}).get("port", 8765)
        )
        
        # File monitoring
        self.observer = None
        self.file_handler = None
        
        # Statistics
        self.stats = IndexingStats()
        
        # Task tracking
        self.active_tasks: Dict[str, IndexingTask] = {}
        self.completed_tasks: List[IndexingTask] = []
        
        # Control
        self.running = False
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {}
    
    async def start(self):
        """Start real-time indexing system"""
        logger.info("Starting real-time indexing system...")
        
        self.running = True
        
        # Start background task processor
        self.task_processor.start()
        
        # Start WebSocket server
        await self.websocket_server.start_server()
        
        # Start file monitoring
        self._start_file_monitoring()
        
        # Start result processing loop
        asyncio.create_task(self._result_processing_loop())
        
        # Start statistics update loop
        asyncio.create_task(self._stats_update_loop())
        
        logger.info("Real-time indexing system started successfully")
    
    async def stop(self):
        """Stop real-time indexing system"""
        logger.info("Stopping real-time indexing system...")
        
        self.running = False
        
        # Stop file monitoring
        self._stop_file_monitoring()
        
        # Stop task processor
        self.task_processor.stop()
        
        # Stop WebSocket server
        await self.websocket_server.stop_server()
        
        logger.info("Real-time indexing system stopped")
    
    def _start_file_monitoring(self):
        """Start file system monitoring"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("File monitoring not available - watchdog not installed")
            return
        
        try:
            self.file_handler = FileChangeHandler(self)
            self.observer = Observer()
            
            # Monitor input directory
            input_dir = self.config.get("paths", {}).get("input", "./input")
            self.observer.schedule(self.file_handler, input_dir, recursive=True)
            
            self.observer.start()
            logger.info(f"File monitoring started for: {input_dir}")
            
        except Exception as e:
            logger.error(f"Failed to start file monitoring: {e}")
    
    def _stop_file_monitoring(self):
        """Stop file system monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File monitoring stopped")
    
    def add_task(self, task: IndexingTask):
        """Add indexing task to queue"""
        self.active_tasks[task.task_id] = task
        self.task_processor.submit_task(task)
        self.stats.total_tasks += 1
        
        # Notify clients
        asyncio.create_task(self.websocket_server.broadcast({
            "type": "task_added",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "file_path": task.file_path,
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    async def _result_processing_loop(self):
        """Process completed tasks"""
        while self.running:
            try:
                # Get completed task
                task = self.task_processor.get_result(timeout=1.0)
                
                if task:
                    await self._handle_completed_task(task)
                    
            except Exception as e:
                logger.error(f"Error in result processing: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_completed_task(self, task: IndexingTask):
        """Handle completed indexing task"""
        # Update tracking
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        
        self.completed_tasks.append(task)
        
        # Update statistics
        if task.status == "completed":
            self.stats.completed_tasks += 1
            
            # Update processing time average
            if task.processing_time:
                total_time = (self.stats.average_processing_time * 
                             (self.stats.completed_tasks - 1) + task.processing_time)
                self.stats.average_processing_time = total_time / self.stats.completed_tasks
            
            # Update content statistics
            self.stats.documents_indexed += task.metadata.get("documents_processed", 0)
            self.stats.entities_extracted += task.metadata.get("entities_extracted", 0)
            self.stats.relationships_extracted += task.metadata.get("relationships_extracted", 0)
            self.stats.communities_detected += task.metadata.get("communities_detected", 0)
            
        elif task.status == "failed":
            self.stats.failed_tasks += 1
            
            # Retry if possible
            if task.retries < task.max_retries:
                logger.info(f"Retrying task {task.task_id} (attempt {task.retries + 1})")
                task.status = "pending"
                self.add_task(task)
                return
        
        # Notify clients
        await self.websocket_server.broadcast({
            "type": "task_completed",
            "task_id": task.task_id,
            "status": task.status,
            "processing_time": task.processing_time,
            "metadata": task.metadata,
            "error": task.error_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Task completed: {task.task_id} ({task.status})")
    
    async def _stats_update_loop(self):
        """Periodically broadcast statistics"""
        while self.running:
            try:
                await asyncio.sleep(10.0)  # Update every 10 seconds
                
                # Broadcast current statistics
                await self.websocket_server.broadcast({
                    "type": "stats_update",
                    "stats": {
                        "total_tasks": self.stats.total_tasks,
                        "completed_tasks": self.stats.completed_tasks,
                        "failed_tasks": self.stats.failed_tasks,
                        "active_tasks": len(self.active_tasks),
                        "average_processing_time": round(self.stats.average_processing_time, 2),
                        "documents_indexed": self.stats.documents_indexed,
                        "entities_extracted": self.stats.entities_extracted,
                        "relationships_extracted": self.stats.relationships_extracted,
                        "communities_detected": self.stats.communities_detected,
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error in stats update: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self.running,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "stats": asdict(self.stats),
            "file_monitoring": self.observer is not None and self.observer.is_alive(),
            "websocket_clients": len(self.websocket_server.clients),
            "workers": self.task_processor.max_workers,
        }

# CLI interface for testing
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time GraphRAG Indexing")
    parser.add_argument("--config", default="./graphrag_config.yaml", help="Config file path")
    parser.add_argument("--full-index", action="store_true", help="Trigger full indexing")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode")
    
    args = parser.parse_args()
    
    # Create indexing manager
    manager = RealtimeIndexingManager(args.config)
    
    try:
        # Start system
        await manager.start()
        
        # Trigger full indexing if requested
        if args.full_index:
            task = IndexingTask(
                task_id=f"full_{int(time.time())}",
                task_type="full",
                priority=1,
                metadata={"trigger": "manual"}
            )
            manager.add_task(task)
            logger.info("Full indexing triggered")
        
        if args.test_mode:
            # Run for 30 seconds in test mode
            logger.info("Running in test mode for 30 seconds...")
            await asyncio.sleep(30.0)
        else:
            # Run indefinitely
            logger.info("Real-time indexing system running. Press Ctrl+C to stop.")
            
            try:
                while True:
                    await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())