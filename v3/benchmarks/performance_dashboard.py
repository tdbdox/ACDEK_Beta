#!/usr/bin/env python
"""
ACDEK v3 Performance Dashboard

This module provides a real-time web dashboard for monitoring ACDEK performance metrics,
visualizing regressions, and managing alerts. It integrates with the continuous_monitor
module to display live performance data.

Features:
1. Real-time performance metric visualization
2. Regression detection and visualization
3. Alert management interface
4. Historical trend analysis
5. Resource usage monitoring
6. Domain-specific performance tracking

Usage:
    python performance_dashboard.py --port 8080
    python continuous_monitor.py --dashboard --dashboard-port 8080
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import socket
import webbrowser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import asyncio
import threading
from enum import Enum

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import relevant components (with try/except to handle potential circular imports)
try:
    from benchmarks.continuous_monitor import ContinuousMonitor, PerformanceMetrics, Alert, AlertLevel
except ImportError:
    # Define stubs for types if imports fail
    class ContinuousMonitor:
        pass
    class PerformanceMetrics:
        pass
    class Alert:
        pass
    class AlertLevel(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_dashboard.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 8080
DEFAULT_UPDATE_INTERVAL = 5  # seconds
DEFAULT_METRICS_COUNT = 20

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ACDEK Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/moment@2.29.4/moment.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #212529;
            line-height: 1.5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            border: 1px solid transparent;
            margin-bottom: -1px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        .tab.active {
            border-color: #dee2e6 #dee2e6 #fff;
            background-color: #fff;
        }
        .tab-content {
            display: none;
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-radius: 0 0 4px 4px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .tab-content.active {
            display: block;
        }
        .metric-card {
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            width: 100%;
            margin-bottom: 20px;
        }
        .alert {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .alert-info {
            background-color: #cce5ff;
            border-color: #b8daff;
            color: #004085;
        }
        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404;
        }
        .alert-error {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        .alert-critical {
            background-color: #dc3545;
            border-color: #bd2130;
            color: #fff;
        }
        .domain-card {
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .domain-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-change {
            font-size: 14px;
        }
        .positive-change {
            color: #28a745;
        }
        .negative-change {
            color: #dc3545;
        }
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-running {
            background-color: #28a745;
        }
        .status-stopped {
            background-color: #dc3545;
        }
        .regression-card {
            background-color: #fff;
            border-left: 4px solid #dc3545;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .regression-moderate {
            border-left-color: #ffc107;
        }
        .regression-severe {
            border-left-color: #dc3545;
        }
        .regression-critical {
            border-left-color: #6f42c1;
            background-color: #f8f9fa;
        }
        .control-panel {
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        button {
            background-color: #007bff;
            color: #fff;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0069d9;
        }
        button.secondary {
            background-color: #6c757d;
        }
        button.secondary:hover {
            background-color: #5a6268;
        }
        button.danger {
            background-color: #dc3545;
        }
        button.danger:hover {
            background-color: #c82333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .last-updated {
            font-size: 12px;
            color: #6c757d;
            text-align: right;
            margin-top: 5px;
        }
        .ws-status {
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 12px;
            background-color: #28a745;
            color: white;
            display: inline-block;
        }
        .ws-status.disconnected {
            background-color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ACDEK Performance Dashboard</h1>
            <div>
                <span id="monitor-status">
                    <span class="status-indicator status-stopped"></span>
                    Monitor Stopped
                </span>
                <span id="ws-status" class="ws-status disconnected">Disconnected</span>
            </div>
        </div>
        
        <div class="control-panel">
            <button id="start-button">Start Monitor</button>
            <button id="stop-button" class="danger">Stop Monitor</button>
            <button id="run-once-button" class="secondary">Run Benchmark</button>
            <span id="last-updated" class="last-updated">Last updated: Never</span>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="overview">Overview</div>
            <div class="tab" data-tab="trends">Performance Trends</div>
            <div class="tab" data-tab="regressions">Regressions</div>
            <div class="tab" data-tab="resources">Resource Usage</div>
            <div class="tab" data-tab="domains">Domains</div>
            <div class="tab" data-tab="alerts">Alerts</div>
        </div>
        
        <div id="overview" class="tab-content active">
            <h2>Performance Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Hallucination Reduction</h3>
                    <div id="hallucination-value" class="metric-value">-</div>
                    <div id="hallucination-change" class="metric-change">-</div>
                </div>
                <div class="metric-card">
                    <h3>Latency Overhead</h3>
                    <div id="latency-value" class="metric-value">-</div>
                    <div id="latency-change" class="metric-change">-</div>
                </div>
                <div class="metric-card">
                    <h3>Confidence Score</h3>
                    <div id="confidence-value" class="metric-value">-</div>
                    <div id="confidence-change" class="metric-change">-</div>
                </div>
                <div class="metric-card">
                    <h3>Success Rate</h3>
                    <div id="success-value" class="metric-value">-</div>
                    <div id="success-change" class="metric-change">-</div>
                </div>
            </div>
            
            <h3>Recent Performance</h3>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
            
            <h3>Latest Regression Summary</h3>
            <div id="regression-summary">
                <p>No regressions detected in the latest run.</p>
            </div>
        </div>
        
        <div id="trends" class="tab-content">
            <h2>Performance Trends</h2>
            <div class="chart-container">
                <canvas id="trend-chart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="domain-trend-chart"></canvas>
            </div>
            <div id="trend-analysis">
                <p>Trend data will appear here after multiple benchmark runs.</p>
            </div>
        </div>
        
        <div id="regressions" class="tab-content">
            <h2>Regression Analysis</h2>
            <div id="regressions-container">
                <p>No regressions have been detected yet.</p>
            </div>
        </div>
        
        <div id="resources" class="tab-content">
            <h2>Resource Usage</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>CPU Usage</h3>
                    <div id="cpu-value" class="metric-value">-</div>
                    <div id="cpu-change" class="metric-change">-</div>
                </div>
                <div class="metric-card">
                    <h3>Memory Usage</h3>
                    <div id="memory-value" class="metric-value">-</div>
                    <div id="memory-change" class="metric-change">-</div>
                </div>
                <div class="metric-card">
                    <h3>Thread Count</h3>
                    <div id="thread-value" class="metric-value">-</div>
                    <div id="thread-change" class="metric-change">-</div>
                </div>
                <div class="metric-card">
                    <h3>Execution Time</h3>
                    <div id="execution-value" class="metric-value">-</div>
                    <div id="execution-change" class="metric-change">-</div>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="resource-chart"></canvas>
            </div>
        </div>
        
        <div id="domains" class="tab-content">
            <h2>Domain Performance</h2>
            <div id="domains-container">
                <p>Domain-specific performance metrics will appear here after benchmark runs.</p>
            </div>
        </div>
        
        <div id="alerts" class="tab-content">
            <h2>Performance Alerts</h2>
            <div id="alerts-container">
                <p>No alerts have been triggered yet.</p>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let socket;
        let performanceChart;
        let trendChart;
        let domainTrendChart;
        let resourceChart;
        let monitorRunning = false;
        let lastMetrics = {};
        let metricsHistory = [];
        let alertsHistory = [];
        
        // Initialize charts
        function initCharts() {
            // Performance chart
            const perfCtx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Hallucination Reduction',
                            data: [],
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            tension: 0.1
                        },
                        {
                            label: 'Latency Overhead (ms)',
                            data: [],
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            tension: 0.1,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Reduction Rate (%)'
                            },
                            ticks: {
                                callback: value => value + '%'
                            }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false
                            },
                            title: {
                                display: true,
                                text: 'Latency (ms)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    }
                }
            });
            
            // Trend chart
            const trendCtx = document.getElementById('trend-chart').getContext('2d');
            trendChart = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Hallucination Reduction',
                            data: [],
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            tension: 0.1
                        },
                        {
                            label: 'Confidence',
                            data: [],
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            tension: 0.1
                        },
                        {
                            label: 'Success Rate',
                            data: [],
                            borderColor: 'rgba(153, 102, 255, 1)',
                            backgroundColor: 'rgba(153, 102, 255, 0.2)',
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Rate (%)'
                            },
                            ticks: {
                                callback: value => value + '%'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Run'
                            }
                        }
                    }
                }
            });
            
            // Domain trend chart
            const domainTrendCtx = document.getElementById('domain-trend-chart').getContext('2d');
            domainTrendChart = new Chart(domainTrendCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Hallucination Reduction by Domain',
                            data: [],
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.7)',
                                'rgba(54, 162, 235, 0.7)',
                                'rgba(255, 206, 86, 0.7)',
                                'rgba(75, 192, 192, 0.7)',
                                'rgba(153, 102, 255, 0.7)'
                            ],
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Reduction Rate (%)'
                            },
                            ticks: {
                                callback: value => value + '%'
                            }
                        }
                    }
                }
            });
            
            // Resource usage chart
            const resourceCtx = document.getElementById('resource-chart').getContext('2d');
            resourceChart = new Chart(resourceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU Usage (%)',
                            data: [],
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            tension: 0.1
                        },
                        {
                            label: 'Memory Usage (MB)',
                            data: [],
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            tension: 0.1,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'CPU (%)'
                            },
                            max: 100
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false
                            },
                            title: {
                                display: true,
                                text: 'Memory (MB)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    }
                }
            });
        }
        
        // Create and connect WebSocket
        function connectWebSocket() {
            socket = new WebSocket(`ws://${window.location.host}/ws`);
            
            socket.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('ws-status').textContent = 'Connected';
                document.getElementById('ws-status').classList.remove('disconnected');
                
                // Request initial data
                socket.send(JSON.stringify({ type: 'get_status' }));
                socket.send(JSON.stringify({ type: 'get_metrics' }));
                socket.send(JSON.stringify({ type: 'get_alerts' }));
            };
            
            socket.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('ws-status').textContent = 'Disconnected';
                document.getElementById('ws-status').classList.add('disconnected');
                
                // Try to reconnect after a delay
                setTimeout(connectWebSocket, 5000);
            };
            
            socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            socket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                switch (message.type) {
                    case 'status':
                        updateStatus(message.data);
                        break;
                    case 'metrics':
                        updateMetrics(message.data);
                        break;
                    case 'metrics_history':
                        updateMetricsHistory(message.data);
                        break;
                    case 'alerts':
                        updateAlerts(message.data);
                        break;
                    case 'regression':
                        updateRegressions(message.data);
                        break;
                }
                
                document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
            };
        }
        
        // Update monitor status
        function updateStatus(status) {
            monitorRunning = status.running;
            
            const statusEl = document.getElementById('monitor-status');
            const indicator = statusEl.querySelector('.status-indicator');
            
            if (monitorRunning) {
                statusEl.innerHTML = '<span class="status-indicator status-running"></span> Monitor Running';
                document.getElementById('start-button').disabled = true;
                document.getElementById('stop-button').disabled = false;
            } else {
                statusEl.innerHTML = '<span class="status-indicator status-stopped"></span> Monitor Stopped';
                document.getElementById('start-button').disabled = false;
                document.getElementById('stop-button').disabled = true;
            }
        }
        
        // Update metrics display
        function updateMetrics(metrics) {
            if (!metrics) return;
            
            lastMetrics = metrics;
            
            // Add to history
            if (metricsHistory.length === 0 || 
                metricsHistory[metricsHistory.length - 1].run_id !== metrics.run_id) {
                metricsHistory.push(metrics);
                
                // Limit history size
                if (metricsHistory.length > 20) {
                    metricsHistory.shift();
                }
                
                // Update charts with new data
                updateCharts();
            }
            
            // Update overview metrics
            updateOverviewMetrics(metrics);
            
            // Update resource metrics
            updateResourceMetrics(metrics);
            
            // Update domain metrics
            updateDomainMetrics(metrics);
            
            // Update regression summary
            if (metrics.regression_metrics) {
                updateRegressionSummary(metrics.regression_metrics);
            }
        }
        
        // Update metrics history
        function updateMetricsHistory(history) {
            metricsHistory = history;
            updateCharts();
        }
        
        // Update overview metrics
        function updateOverviewMetrics(metrics) {
            // Hallucination reduction
            if (metrics.metrics && 'hallucination_reduction' in metrics.metrics) {
                const value = metrics.metrics.hallucination_reduction;
                document.getElementById('hallucination-value').textContent = 
                    (value * 100).toFixed(2) + '%';
                
                if (metrics.regression_metrics && 'hallucination_reduction' in metrics.regression_metrics) {
                    const change = metrics.regression_metrics.hallucination_reduction;
                    const delta = change.delta;
                    const pctChange = (delta * 100).toFixed(2);
                    const changeEl = document.getElementById('hallucination-change');
                    
                    if (Math.abs(delta) < 0.0001) {
                        changeEl.textContent = 'No change';
                        changeEl.className = 'metric-change';
                    } else if (delta > 0) {
                        changeEl.textContent = '▲ ' + pctChange + '% (improved)';
                        changeEl.className = 'metric-change positive-change';
                    } else {
                        changeEl.textContent = '▼ ' + Math.abs(pctChange) + '% (worse)';
                        changeEl.className = 'metric-change negative-change';
                    }
                }
            }
            
            // Latency overhead
            if (metrics.metrics && 'latency_overhead' in metrics.metrics) {
                const value = metrics.metrics.latency_overhead;
                document.getElementById('latency-value').textContent = 
                    value.toFixed(2) + ' ms';
                
                if (metrics.regression_metrics && 'latency_overhead' in metrics.regression_metrics) {
                    const change = metrics.regression_metrics.latency_overhead;
                    const delta = change.delta;
                    const pctChange = (delta / Math.max(0.1, Math.abs(change.baseline)) * 100).toFixed(2);
                    const changeEl = document.getElementById('latency-change');
                    
                    if (Math.abs(delta) < 0.1) {
                        changeEl.textContent = 'No change';
                        changeEl.className = 'metric-change';
                    } else if (delta < 0) {
                        changeEl.textContent = '▼ ' + Math.abs(pctChange) + '% (improved)';
                        changeEl.className = 'metric-change positive-change';
                    } else {
                        changeEl.textContent = '▲ ' + pctChange + '% (worse)';
                        changeEl.className = 'metric-change negative-change';
                    }
                }
            }
            
            // Confidence
            if (metrics.metrics && 'confidence' in metrics.metrics) {
                const value = metrics.metrics.confidence;
                document.getElementById('confidence-value').textContent = 
                    (value * 100).toFixed(2) + '%';
                
                if (metrics.regression_metrics && 'confidence' in metrics.regression_metrics) {
                    const change = metrics.regression_metrics.confidence;
                    const delta = change.delta;
                    const pctChange = (delta * 100).toFixed(2);
                    const changeEl = document.getElementById('confidence-change');
                    
                    if (Math.abs(delta) < 0.0001) {
                        changeEl.textContent = 'No change';
                        changeEl.className = 'metric-change';
                    } else if (delta > 0) {
                        changeEl.textContent = '▲ ' + pctChange + '% (improved)';
                        changeEl.className = 'metric-change positive-change';
                    } else {
                        changeEl.textContent = '▼ ' + Math.abs(pctChange) + '% (worse)';
                        changeEl.className = 'metric-change negative-change';
                    }
                }
            }
            
            // Success rate
            if (metrics.metrics && 'success_rate' in metrics.metrics) {
                const value = metrics.metrics.success_rate;
                document.getElementById('success-value').textContent = 
                    (value * 100).toFixed(2) + '%';
                
                if (metrics.regression_metrics && 'success_rate' in metrics.regression_metrics) {
                    const change = metrics.regression_metrics.success_rate;
                    const delta = change.delta;
                    const pctChange = (delta * 100).toFixed(2);
                    const changeEl = document.getElementById('success-change');
                    
                    if (Math.abs(delta) < 0.0001) {
                        changeEl.textContent = 'No change';
                        changeEl.className = 'metric-change';
                    } else if (delta > 0) {
                        changeEl.textContent = '▲ ' + pctChange + '% (improved)';
                        changeEl.className = 'metric-change positive-change';
                    } else {
                        changeEl.textContent = '▼ ' + Math.abs(pctChange) + '% (worse)';
                        changeEl.className = 'metric-change negative-change';
                    }
                }
            }
        }
        
        // Update resource metrics
        function updateResourceMetrics(metrics) {
            if (!metrics.resource_metrics) return;
            
            // CPU usage
            if ('avg_cpu_percent' in metrics.resource_metrics) {
                const value = metrics.resource_metrics.avg_cpu_percent;
                document.getElementById('cpu-value').textContent = 
                    value.toFixed(2) + '%';
                
                // Add change display if available
            }
            
            // Memory usage
            if ('avg_memory_mb' in metrics.resource_metrics) {
                const value = metrics.resource_metrics.avg_memory_mb;
                document.getElementById('memory-value').textContent = 
                    value.toFixed(2) + ' MB';
                
                // Add change display if available
            }
            
            // Thread count
            if ('avg_thread_count' in metrics.resource_metrics) {
                const value = metrics.resource_metrics.avg_thread_count;
                document.getElementById('thread-value').textContent = 
                    Math.round(value);
                
                // Add change display if available
            }
            
            // Execution time
            if (metrics.metrics && 'execution_time' in metrics.metrics) {
                const value = metrics.metrics.execution_time;
                document.getElementById('execution-value').textContent = 
                    value.toFixed(2) + ' sec';
                
                if (metrics.regression_metrics && 'execution_time' in metrics.regression_metrics) {
                    const change = metrics.regression_metrics.execution_time;
                    const delta = change.delta;
                    const pctChange = (delta / Math.max(0.1, Math.abs(change.baseline)) * 100).toFixed(2);
                    const changeEl = document.getElementById('execution-change');
                    
                    if (Math.abs(delta) < 0.1) {
                        changeEl.textContent = 'No change';
                        changeEl.className = 'metric-change';
                    } else if (delta < 0) {
                        changeEl.textContent = '▼ ' + Math.abs(pctChange) + '% (improved)';
                        changeEl.className = 'metric-change positive-change';
                    } else {
                        changeEl.textContent = '▲ ' + pctChange + '% (worse)';
                        changeEl.className = 'metric-change negative-change';
                    }
                }
            }
        }
        
        // Update domain metrics
        function updateDomainMetrics(metrics) {
            if (!metrics.domain_metrics) return;
            
            const domainsContainer = document.getElementById('domains-container');
            domainsContainer.innerHTML = '';
            
            for (const domain in metrics.domain_metrics) {
                const domainMetrics = metrics.domain_metrics[domain];
                
                const domainCard = document.createElement('div');
                domainCard.className = 'domain-card';
                
                const title = document.createElement('h3');
                title.textContent = domain.charAt(0).toUpperCase() + domain.slice(1);
                domainCard.appendChild(title);
                
                const metricsDiv = document.createElement('div');
                metricsDiv.className = 'domain-metrics';
                
                // Add hallucination reduction
                if ('hallucination_reduction' in domainMetrics) {
                    const metricDiv = document.createElement('div');
                    
                    const metricName = document.createElement('div');
                    metricName.textContent = 'Hallucination Reduction';
                    
                    const metricValue = document.createElement('div');
                    metricValue.className = 'metric-value';
                    metricValue.textContent = (domainMetrics.hallucination_reduction * 100).toFixed(2) + '%';
                    
                    metricDiv.appendChild(metricName);
                    metricDiv.appendChild(metricValue);
                    metricsDiv.appendChild(metricDiv);
                }
                
                // Add latency overhead
                if ('latency_overhead' in domainMetrics) {
                    const metricDiv = document.createElement('div');
                    
                    const metricName = document.createElement('div');
                    metricName.textContent = 'Latency Overhead';
                    
                    const metricValue = document.createElement('div');
                    metricValue.className = 'metric-value';
                    metricValue.textContent = domainMetrics.latency_overhead.toFixed(2) + ' ms';
                    
                    metricDiv.appendChild(metricName);
                    metricDiv.appendChild(metricValue);
                    metricsDiv.appendChild(metricDiv);
                }
                
                domainCard.appendChild(metricsDiv);
                domainsContainer.appendChild(domainCard);
            }
            
            if (Object.keys(metrics.domain_metrics).length === 0) {
                domainsContainer.innerHTML = '<p>No domain-specific metrics available.</p>';
            }
        }
        
        // Update all charts
        function updateCharts() {
            if (metricsHistory.length === 0) return;
            
            // Update performance chart
            updatePerformanceChart();
            
            // Update trend chart
            updateTrendChart();
            
            // Update domain trend chart
            updateDomainTrendChart();
            
            // Update resource chart
            updateResourceChart();
        }
        
        // Update performance chart
        function updatePerformanceChart() {
            const labels = metricsHistory.map(m => {
                const timestamp = m.timestamp ? new Date(m.timestamp) : null;
                return timestamp ? timestamp.toLocaleTimeString() : `Run ${m.run_id.slice(0, 8)}`;
            });
            
            const hallucinationData = metricsHistory.map(m => {
                if (m.metrics && 'hallucination_reduction' in m.metrics) {
                    return m.metrics.hallucination_reduction * 100;
                }
                return null;
            });
            
            const latencyData = metricsHistory.map(m => {
                if (m.metrics && 'latency_overhead' in m.metrics) {
                    return m.metrics.latency_overhead;
                }
                return null;
            });
            
            performanceChart.data.labels = labels;
            performanceChart.data.datasets[0].data = hallucinationData;
            performanceChart.data.datasets[1].data = latencyData;
            performanceChart.update();
        }
        
        // Update trend chart
        function updateTrendChart() {
            const labels = metricsHistory.map(m => {
                return m.run_id.slice(0, 8);
            });
            
            const hallucinationData = metricsHistory.map(m => {
                if (m.metrics && 'hallucination_reduction' in m.metrics) {
                    return m.metrics.hallucination_reduction * 100;
                }
                return null;
            });
            
            const confidenceData = metricsHistory.map(m => {
                if (m.metrics && 'confidence' in m.metrics) {
                    return m.metrics.confidence * 100;
                }
                return null;
            });
            
            const successData = metricsHistory.map(m => {
                if (m.metrics && 'success_rate' in m.metrics) {
                    return m.metrics.success_rate * 100;
                }
                return null;
            });
            
            trendChart.data.labels = labels;
            trendChart.data.datasets[0].data = hallucinationData;
            trendChart.data.datasets[1].data = confidenceData;
            trendChart.data.datasets[2].data = successData;
            trendChart.update();
            
            // Update trend analysis text
            const trendAnalysis = document.getElementById('trend-analysis');
            
            if (metricsHistory.length < 3) {
                trendAnalysis.innerHTML = '<p>Trend analysis requires at least 3 data points.</p>';
                return;
            }
            
            // Calculate simple trend for hallucination reduction
            const firstHall = hallucinationData[0];
            const lastHall = hallucinationData[hallucinationData.length - 1];
            const diffHall = lastHall - firstHall;
            
            // Calculate simple trend for latency
            const firstLat = latencyData[0];
            const lastLat = latencyData[latencyData.length - 1];
            const diffLat = lastLat - firstLat;
            
            let hallTrend = 'stable';
            if (diffHall > 5) hallTrend = 'improving';
            else if (diffHall < -5) hallTrend = 'degrading';
            
            let latTrend = 'stable';
            if (diffLat < -5) latTrend = 'improving';
            else if (diffLat > 5) latTrend = 'degrading';
            
            trendAnalysis.innerHTML = `
                <h3>Trend Summary</h3>
                <p>Hallucination Reduction: <strong>${hallTrend}</strong> (${diffHall > 0 ? '+' : ''}${diffHall.toFixed(2)}% over ${metricsHistory.length} runs)</p>
                <p>Latency Overhead: <strong>${latTrend}</strong> (${diffLat > 0 ? '+' : ''}${diffLat.toFixed(2)}ms over ${metricsHistory.length} runs)</p>
                <p>Analysis period: ${new Date(metricsHistory[0].timestamp).toLocaleString()} to ${new Date(metricsHistory[metricsHistory.length-1].timestamp).toLocaleString()}</p>
            `;
        }
        
        // Update domain trend chart
        function updateDomainTrendChart() {
            // Get the latest metrics with domain data
            let latestWithDomains = null;
            for (let i = metricsHistory.length - 1; i >= 0; i--) {
                if (metricsHistory[i].domain_metrics && Object.keys(metricsHistory[i].domain_metrics).length > 0) {
                    latestWithDomains = metricsHistory[i];
                    break;
                }
            }
            
            if (!latestWithDomains) return;
            
            const domains = Object.keys(latestWithDomains.domain_metrics);
            const hallucinationData = domains.map(domain => {
                const domainMetrics = latestWithDomains.domain_metrics[domain];
                if (domainMetrics && 'hallucination_reduction' in domainMetrics) {
                    return domainMetrics.hallucination_reduction * 100;
                }
                return 0;
            });
            
            domainTrendChart.data.labels = domains.map(d => d.charAt(0).toUpperCase() + d.slice(1));
            domainTrendChart.data.datasets[0].data = hallucinationData;
            domainTrendChart.update();
        }
        
        // Update resource chart
        function updateResourceChart() {
            const labels = metricsHistory.map(m => {
                const timestamp = m.timestamp ? new Date(m.timestamp) : null;
                return timestamp ? timestamp.toLocaleTimeString() : `Run ${m.run_id.slice(0, 8)}`;
            });
            
            const cpuData = metricsHistory.map(m => {
                if (m.resource_metrics && 'avg_cpu_percent' in m.resource_metrics) {
                    return m.resource_metrics.avg_cpu_percent;
                }
                return null;
            });
            
            const memoryData = metricsHistory.map(m => {
                if (m.resource_metrics && 'avg_memory_mb' in m.resource_metrics) {
                    return m.resource_metrics.avg_memory_mb;
                }
                return null;
            });
            
            resourceChart.data.labels = labels;
            resourceChart.data.datasets[0].data = cpuData;
            resourceChart.data.datasets[1].data = memoryData;
            resourceChart.update();
        }
        
        // Update alerts display
        function updateAlerts(alerts) {
            if (!alerts || alerts.length === 0) return;
            
            alertsHistory = alerts;
            
            const alertsContainer = document.getElementById('alerts-container');
            alertsContainer.innerHTML = '';
            
            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${alert.level}`;
                
                const timestamp = new Date(alert.timestamp).toLocaleString();
                
                alertDiv.innerHTML = `
                    <h4>${alert.message}</h4>
                    <p><strong>Source:</strong> ${alert.source}</p>
                    <p><strong>Time:</strong> ${timestamp}</p>
                `;
                
                if (alert.metrics && Object.keys(alert.metrics).length > 0) {
                    const metricsList = document.createElement('ul');
                    
                    for (const key in alert.metrics) {
                        const item = document.createElement('li');
                        item.textContent = `${key}: ${alert.metrics[key]}`;
                        metricsList.appendChild(item);
                    }
                    
                    alertDiv.appendChild(metricsList);
                }
                
                alertsContainer.appendChild(alertDiv);
            });
            
            if (alerts.length === 0) {
                alertsContainer.innerHTML = '<p>No alerts have been triggered yet.</p>';
            }
        }
        
        // Update regressions display
        function updateRegressions(regressions) {
            if (!regressions) return;
            
            const container = document.getElementById('regressions-container');
            container.innerHTML = '';
            
            if (!regressions.regression_metrics || Object.keys(regressions.regression_metrics).length === 0) {
                container.innerHTML = '<p>No regressions have been detected yet.</p>';
                return;
            }
            
            const regressionsList = [];
            
            for (const metric in regressions.regression_metrics) {
                const reg = regressions.regression_metrics[metric];
                
                if (reg.is_regression) {
                    regressionsList.push({
                        name: metric,
                        current: reg.current,
                        baseline: reg.baseline,
                        delta: reg.delta,
                        percent: Math.abs(reg.delta / Math.max(0.0001, Math.abs(reg.baseline)) * 100),
                        domain: reg.domain
                    });
                }
            }
            
            regressionsList.sort((a, b) => b.percent - a.percent);
            
            regressionsList.forEach(reg => {
                const card = document.createElement('div');
                
                // Determine severity
                let severityClass = '';
                let severityText = '';
                
                if (reg.percent >= 20) {
                    severityClass = 'regression-critical';
                    severityText = 'CRITICAL';
                } else if (reg.percent >= 10) {
                    severityClass = 'regression-severe';
                    severityText = 'SEVERE';
                } else if (reg.percent >= 5) {
                    severityClass = 'regression-moderate';
                    severityText = 'MODERATE';
                } else {
                    severityClass = 'regression-minor';
                    severityText = 'MINOR';
                }
                
                card.className = `regression-card ${severityClass}`;
                
                let context = '';
                if (reg.domain) {
                    context = `in ${reg.domain} domain`;
                }
                
                let valueFormatted = (metric) => {
                    if (metric.name.includes('reduction') || metric.name.includes('confidence') || 
                        metric.name.includes('rate')) {
                        return (metric.current * 100).toFixed(2) + '%';
                    }
                    if (metric.name.includes('latency') || metric.name.includes('overhead')) {
                        return metric.current.toFixed(2) + ' ms';
                    }
                    return metric.current.toFixed(4);
                };
                
                card.innerHTML = `
                    <h3>${reg.name} ${context} <span style="float:right">${severityText}</span></h3>
                    <p><strong>Value:</strong> ${valueFormatted(reg)} (was ${reg.baseline.toFixed(4)})</p>
                    <p><strong>Change:</strong> ${reg.delta > 0 ? '+' : ''}${reg.delta.toFixed(4)} (${reg.percent.toFixed(2)}%)</p>
                `;
                
                container.appendChild(card);
            });
            
            if (regressionsList.length === 0) {
                container.innerHTML = '<p>No regressions detected in the latest benchmark run.</p>';
            }
        }
        
        // Update regression summary
        function updateRegressionSummary(regressions) {
            if (!regressions) return;
            
            const container = document.getElementById('regression-summary');
            
            let regressionCount = 0;
            let severeCount = 0;
            
            for (const metric in regressions) {
                const reg = regressions[metric];
                if (reg.is_regression) {
                    regressionCount++;
                    
                    const percent = Math.abs(reg.delta / Math.max(0.0001, Math.abs(reg.baseline)) * 100);
                    if (percent >= 10) {
                        severeCount++;
                    }
                }
            }
            
            if (regressionCount === 0) {
                container.innerHTML = '<p>No regressions detected in the latest run.</p>';
                return;
            }
            
            container.innerHTML = `
                <p>Detected ${regressionCount} regressions in the latest run, including ${severeCount} severe regressions.</p>
                <button onclick="document.querySelector('.tab[data-tab=\\'regressions\\']').click()">View Details</button>
            `;
        }
        
        // Send command to server
        function sendCommand(command) {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: command }));
            }
        }
        
        // Initialize tabs
        function initTabs() {
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs and contents
                    tabs.forEach(t => t.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab and corresponding content
                    tab.classList.add('active');
                    const tabName = tab.getAttribute('data-tab');
                    document.getElementById(tabName).classList.add('active');
                    
                    // Refresh charts when tab is activated
                    if (tabName === 'trends' || tabName === 'resources') {
                        updateCharts();
                    }
                });
            });
        }
        
        // Initialize controls
        function initControls() {
            document.getElementById('start-button').addEventListener('click', () => {
                sendCommand('start_monitor');
            });
            
            document.getElementById('stop-button').addEventListener('click', () => {
                sendCommand('stop_monitor');
            });
            
            document.getElementById('run-once-button').addEventListener('click', () => {
                sendCommand('run_once');
            });
        }
        
        // Initialize dashboard
        function init() {
            initTabs();
            initControls();
            initCharts();
            connectWebSocket();
        }
        
        // Start on load
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""

# API models
class StatusResponse(BaseModel):
    running: bool
    monitor_pid: Optional[int] = None
    uptime: Optional[float] = None
    next_run: Optional[str] = None

class CommandRequest(BaseModel):
    command: str

class MetricsResponse(BaseModel):
    metrics: List[Dict[str, Any]]

# WebSocket manager
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

# FastAPI app
app = FastAPI(title="ACDEK Performance Dashboard")

# Cross Origin Resource Sharing settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSockets
manager = ConnectionManager()

# Global monitor reference
monitor = None
monitor_start_time = None

# Background task for periodic updates
async def send_periodic_updates():
    """Background task that sends periodic updates to all connected clients."""
    while True:
        if monitor is not None and manager.active_connections:
            try:
                # Get latest metrics
                latest_metrics = monitor.get_latest_metrics()
                if latest_metrics:
                    # Broadcast to all clients
                    await manager.broadcast({
                        "type": "metrics",
                        "data": latest_metrics.to_dict() if hasattr(latest_metrics, "to_dict") else latest_metrics
                    })
                
                # Send monitor status
                await manager.broadcast({
                    "type": "status",
                    "data": {
                        "running": monitor.running if hasattr(monitor, "running") else False,
                        "uptime": time.time() - monitor_start_time if monitor_start_time else None
                    }
                })
            except Exception as e:
                logger.error(f"Error sending periodic updates: {e}")
        
        # Wait for next update interval
        await asyncio.sleep(DEFAULT_UPDATE_INTERVAL)

# Serve the dashboard HTML
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML."""
    return DASHBOARD_HTML

# API endpoints
@app.get("/api/status")
async def get_status():
    """Get the current status of the performance monitor."""
    if monitor is None:
        return JSONResponse(content={"running": False})
    
    running = hasattr(monitor, "running") and monitor.running
    
    return StatusResponse(
        running=running,
        uptime=time.time() - monitor_start_time if monitor_start_time else None,
        next_run=None  # TODO: Calculate next run time
    )

@app.post("/api/command")
async def send_command(command: CommandRequest, background_tasks: BackgroundTasks):
    """Send a command to the performance monitor."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    global monitor_start_time
    
    if command.command == "start":
        if not (hasattr(monitor, "running") and monitor.running):
            monitor.start()
            monitor_start_time = time.time()
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
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command.command}")

@app.get("/api/metrics")
async def get_metrics(count: int = Query(DEFAULT_METRICS_COUNT, ge=1, le=100)):
    """Get recent metrics."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    try:
        metrics = monitor.get_recent_metrics(count=count)
        return {
            "metrics": [m.to_dict() if hasattr(m, "to_dict") else m for m in metrics]
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
async def get_alerts(count: int = Query(10, ge=1, le=100), level: Optional[str] = None):
    """Get recent alerts."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    try:
        alerts = monitor.get_recent_alerts(count=count, level=level)
        return {
            "alerts": [a.to_dict() if hasattr(a, "to_dict") else a for a in alerts]
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from the client
            message_text = await websocket.receive_text()
            message = json.loads(message_text)
            
            # Handle different message types
            if message.get("type") == "get_status":
                # Send monitor status
                await manager.send_personal_message({
                    "type": "status",
                    "data": {
                        "running": monitor.running if monitor and hasattr(monitor, "running") else False,
                        "uptime": time.time() - monitor_start_time if monitor_start_time else None
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
                    recent_metrics = monitor.get_recent_metrics(count=DEFAULT_METRICS_COUNT)
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
            
            elif message.get("type") == "start_monitor":
                # Start the monitor
                if monitor and not (hasattr(monitor, "running") and monitor.running):
                    monitor.start()
                    monitor_start_time = time.time()
                    await manager.broadcast({
                        "type": "status",
                        "data": {"running": True}
                    })
            
            elif message.get("type") == "stop_monitor":
                # Stop the monitor
                if monitor and hasattr(monitor, "running") and monitor.running:
                    monitor.stop()
                    await manager.broadcast({
                        "type": "status",
                        "data": {"running": False}
                    })
            
            elif message.get("type") == "run_once":
                # Run a benchmark once
                if monitor:
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

def run_dashboard(monitor_instance: ContinuousMonitor = None, port: int = DEFAULT_PORT):
    """
    Run the dashboard server.
    
    Args:
        monitor_instance: ContinuousMonitor instance to connect to
        port: Port to run the server on
    """
    global monitor, monitor_start_time
    
    # Set the monitor instance
    monitor = monitor_instance
    if monitor and hasattr(monitor, "running") and monitor.running:
        monitor_start_time = time.time()
    
    # Create a config for uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    
    # Start background task for periodic updates
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(send_periodic_updates())
    
    # Find an available port if the specified port is in use
    while is_port_in_use(port):
        logger.warning(f"Port {port} is in use, trying {port+1}")
        port += 1
        config.port = port
    
    logger.info(f"Starting dashboard at http://localhost:{port}")
    
    # Try to open the dashboard in a browser
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        logger.info(f"Unable to open browser automatically. Please visit http://localhost:{port}")
    
    # Start the server
    server = uvicorn.Server(config)
    server.run()

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACDEK v3 Performance Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to run the dashboard on"
    )
    
    parser.add_argument(
        "--monitor-config",
        type=str,
        help="Path to monitor configuration file (to start a monitor)"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create a monitor if config is provided
    if args.monitor_config:
        try:
            from benchmarks.continuous_monitor import ContinuousMonitor
            logger.info(f"Creating monitor with config {args.monitor_config}")
            
            monitor_instance = ContinuousMonitor(config_path=args.monitor_config)
            run_dashboard(monitor_instance=monitor_instance, port=args.port)
        except ImportError:
            logger.error("Failed to import ContinuousMonitor. Running dashboard without monitor.")
            run_dashboard(port=args.port)
    else:
        # Run without a monitor
        run_dashboard(port=args.port)

if __name__ == "__main__":
    main()