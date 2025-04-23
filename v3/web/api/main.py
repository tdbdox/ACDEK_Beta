"""
FastAPI backend for ACDEK Executive Suite telemetry, performance monitoring, 
and pipeline endpoints.

This module provides REST API and WebSocket endpoints for:
1. Running the ACDEK pipeline
2. Monitoring performance metrics
3. Receiving real-time telemetry data
4. Managing performance benchmarks
5. Visualizing performance data
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import sys
import os
import logging
from pathlib import Path
import time
import asyncio

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import ACDEK components
try:
    from acdek.acdek_kernel import ACDEKKernel
    from benchmarks.continuous_monitor import ContinuousMonitor, PerformanceMetrics, Alert
    from benchmarks.regression_detector import RegressionDetector
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    # Create stub classes if imports fail
    class ContinuousMonitor:
        def __init__(self, *args, **kwargs):
            self.running = False
        def get_latest_metrics(self):
            return {}
        def get_recent_metrics(self, count=10):
            return []
        def get_recent_alerts(self, count=10, level=None):
            return []
        def start(self):
            self.running = True
        def stop(self):
            self.running = False
        def run_once(self):
            return {"run_id": "mock", "metrics": {}}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="ACDEK API Server")

# Cross Origin Resource Sharing settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global monitor reference
monitor = None
if MONITOR_AVAILABLE:
    try:
        # Create monitor with default settings
        monitor = ContinuousMonitor()
        logger.info("Performance monitor initialized")
    except Exception as e:
        logger.error(f"Error initializing monitor: {e}")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection might be closed
                pass

# Create connection manager
manager = ConnectionManager()

# Pydantic models
class RunRequest(BaseModel):
    data: dict

class CommandRequest(BaseModel):
    command: str
    params: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    running: bool
    monitor_available: bool
    uptime: Optional[float] = None
    next_run: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    # Start background task for periodic updates if monitor is available
    if monitor is not None:
        asyncio.create_task(send_periodic_updates())

# Background task for periodic updates
async def send_periodic_updates():
    """Background task that sends periodic updates to connected clients."""
    while True:
        if monitor is not None and manager.active_connections:
            try:
                # Get latest metrics
                latest_metrics = monitor.get_latest_metrics()
                if latest_metrics:
                    # Convert to dict if it's an object
                    metrics_dict = latest_metrics.to_dict() if hasattr(latest_metrics, "to_dict") else latest_metrics
                    
                    # Broadcast to all clients
                    await manager.broadcast({
                        "type": "metrics",
                        "data": metrics_dict
                    })
                
                # Send monitor status
                await manager.broadcast({
                    "type": "status",
                    "data": {
                        "running": monitor.running if hasattr(monitor, "running") else False
                    }
                })
            except Exception as e:
                logger.error(f"Error sending periodic updates: {e}")
        
        # Wait for next update interval
        await asyncio.sleep(5)  # 5 second update interval

# API endpoints
@app.post("/run")
async def run_pipeline(request: RunRequest):
    """
    Run the ACDEK pipeline with the provided data.
    
    This endpoint executes the full ACDEK pipeline including:
    - Hallucination detection
    - Constraint verification
    - Confidence scoring
    - Performance profiling
    """
    try:
        kernel = ACDEKKernel()
        start_time = time.time()
        result = kernel.run(request.data)
        execution_time = time.time() - start_time
        
        # Add performance metadata
        result["execution_time"] = execution_time
        result["timestamp"] = time.time()
        
        return {"result": result}
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate_constraints(request: RunRequest):
    """Validate constraints on the input data."""
    try:
        from enrichment.constraint_verifier import ConstraintVerifier
        verifier = ConstraintVerifier()
        result = verifier.validate(request.data)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error validating constraints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confidence")
async def score_confidence(request: RunRequest):
    """Calculate confidence scores for the input data."""
    try:
        from enrichment.confidence_scorer import VectorConfidenceScorer
        scorer = VectorConfidenceScorer()
        score = scorer.score(request.data)
        return {
            "aggregate_score": score.aggregate_score,
            "factors": {f.value if hasattr(f, "value") else str(f): v for f, v in score.factors.items()},
            "metadata": {k: v for k, v in score.metadata.items() if k != "vector"}
        }
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get the current status of the API server and performance monitor."""
    return StatusResponse(
        running=monitor.running if monitor and hasattr(monitor, "running") else False,
        monitor_available=MONITOR_AVAILABLE,
        uptime=None  # TODO: Track uptime
    )

@app.post("/monitor/command")
async def send_monitor_command(command: CommandRequest, background_tasks: BackgroundTasks):
    """
    Send a command to the performance monitor.
    
    Commands:
    - start: Start continuous monitoring
    - stop: Stop continuous monitoring
    - run_once: Run a single benchmark
    - set_baseline: Set the current run as baseline
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not available")
    
    if command.command == "start":
        if not (hasattr(monitor, "running") and monitor.running):
            monitor.start()
            background_tasks.add_task(lambda: manager.broadcast({
                "type": "status",
                "data": {"running": True}
            }))
            return {"status": "started"}
        return {"status": "already_running"}
    
    elif command.command == "stop":
        if hasattr(monitor, "running") and monitor.running:
            monitor.stop()
            background_tasks.add_task(lambda: manager.broadcast({
                "type": "status",
                "data": {"running": False}
            }))
            return {"status": "stopped"}
        return {"status": "not_running"}
    
    elif command.command == "run_once":
        result = monitor.run_once()
        background_tasks.add_task(lambda: manager.broadcast({
            "type": "metrics",
            "data": result.get("metrics", {})
        }))
        return {"status": "benchmark_executed", "run_id": result.get("run_id")}
    
    elif command.command == "set_baseline":
        # TODO: Implement set_baseline
        return {"status": "not_implemented"}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command.command}")

@app.get("/metrics")
async def get_metrics(count: int = Query(10, ge=1, le=100)):
    """Get recent performance metrics."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not available")
    
    try:
        metrics = monitor.get_recent_metrics(count=count)
        return {
            "metrics": [m.to_dict() if hasattr(m, "to_dict") else m for m in metrics]
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts(count: int = Query(10, ge=1, le=100), level: Optional[str] = None):
    """Get recent performance alerts."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not available")
    
    try:
        alerts = monitor.get_recent_alerts(count=count, level=level)
        return {
            "alerts": [a.to_dict() if hasattr(a, "to_dict") else a for a in alerts]
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial status
        await manager.send_personal_message({
            "type": "status",
            "data": {
                "running": monitor.running if monitor and hasattr(monitor, "running") else False,
                "monitor_available": MONITOR_AVAILABLE
            }
        }, websocket)
        
        while True:
            # Wait for messages from the client
            message_text = await websocket.receive_text()
            message = json.loads(message_text)
            
            # Handle client messages
            if message.get("type") == "get_status":
                # Send monitor status
                await manager.send_personal_message({
                    "type": "status",
                    "data": {
                        "running": monitor.running if monitor and hasattr(monitor, "running") else False,
                        "monitor_available": MONITOR_AVAILABLE
                    }
                }, websocket)
            
            elif message.get("type") == "get_metrics":
                # Send latest metrics
                if monitor:
                    latest_metrics = monitor.get_latest_metrics()
                    if latest_metrics:
                        await manager.send_personal_message({
                            "type": "metrics",
                            "data": latest_metrics.to_dict() if hasattr(latest_metrics, "to_dict") else latest_metrics
                        }, websocket)
                    
                    # Also send historical metrics
                    recent_metrics = monitor.get_recent_metrics(count=20)
                    await manager.send_personal_message({
                        "type": "metrics_history",
                        "data": [m.to_dict() if hasattr(m, "to_dict") else m for m in recent_metrics]
                    }, websocket)
            
            elif message.get("type") == "get_alerts":
                # Send alerts
                if monitor:
                    alerts = monitor.get_recent_alerts(count=10)
                    await manager.send_personal_message({
                        "type": "alerts",
                        "data": [a.to_dict() if hasattr(a, "to_dict") else a for a in alerts]
                    }, websocket)
            
            elif message.get("type") == "command":
                # Process command
                command = message.get("command")
                
                if command == "start_monitor" and monitor:
                    if not (hasattr(monitor, "running") and monitor.running):
                        monitor.start()
                        await manager.broadcast({
                            "type": "status",
                            "data": {"running": True}
                        })
                
                elif command == "stop_monitor" and monitor:
                    if hasattr(monitor, "running") and monitor.running:
                        monitor.stop()
                        await manager.broadcast({
                            "type": "status",
                            "data": {"running": False}
                        })
                
                elif command == "run_once" and monitor:
                    result = monitor.run_once()
                    if "metrics" in result:
                        await manager.broadcast({
                            "type": "metrics",
                            "data": result["metrics"]
                        })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)