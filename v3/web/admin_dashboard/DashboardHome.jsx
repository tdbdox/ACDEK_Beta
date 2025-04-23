import React, { useState, useEffect } from 'react';
import SessionMetricsPanel from './SessionMetricsPanel';

/**
 * DashboardHome
 * Main admin dashboard UI shell with real-time performance monitoring integration.
 */
const DashboardHome = () => {
  const [monitorStatus, setMonitorStatus] = useState({
    running: false,
    available: false
  });
  const [metrics, setMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [socket, setSocket] = useState(null);

  // Connect to WebSocket on component mount
  useEffect(() => {
    // Determine WebSocket URL (adjust for production vs development environments)
    const wsUrl = window.location.hostname === 'localhost' 
      ? `ws://${window.location.hostname}:8000/ws` // Dev backend port
      : `ws://${window.location.host}/ws`; // Production

    // Create WebSocket
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsStatus('connected');
      
      // Request initial data
      ws.send(JSON.stringify({ type: 'get_status' }));
      ws.send(JSON.stringify({ type: 'get_metrics' }));
      ws.send(JSON.stringify({ type: 'get_alerts' }));
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsStatus('disconnected');
      
      // Try to reconnect after a delay
      setTimeout(() => {
        setSocket(null); // This will trigger a new connection attempt
      }, 5000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsStatus('error');
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
          case 'status':
            setMonitorStatus({
              running: message.data.running,
              available: message.data.monitor_available
            });
            break;
          case 'metrics':
            setMetrics(message.data);
            break;
          case 'alerts':
            setAlerts(message.data);
            break;
          default:
            console.log('Unknown message type:', message.type);
        }
      } catch (e) {
        console.error('Error processing WebSocket message:', e);
      }
    };
    
    setSocket(ws);
    
    // Clean up on unmount
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [socket === null]); // Reconnect when socket is null

  // Send command to server
  const sendCommand = (command) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ 
        type: 'command',
        command: command
      }));
    } else {
      console.warn('WebSocket not connected, cannot send command');
    }
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>ACDEK Admin Dashboard</h1>
        <div className="connection-status">
          <div className={`status-indicator ${monitorStatus.running ? 'status-running' : 'status-stopped'}`}></div>
          <span>Monitor {monitorStatus.running ? 'Running' : 'Stopped'}</span>
          <div className={`ws-status ${wsStatus}`}>{wsStatus === 'connected' ? 'Connected' : 'Disconnected'}</div>
        </div>
      </header>

      <div className="dashboard-controls">
        <button 
          onClick={() => sendCommand('start_monitor')}
          disabled={!monitorStatus.available || monitorStatus.running}
        >
          Start Monitor
        </button>
        <button 
          onClick={() => sendCommand('stop_monitor')}
          disabled={!monitorStatus.available || !monitorStatus.running}
          className="danger"
        >
          Stop Monitor
        </button>
        <button 
          onClick={() => sendCommand('run_once')}
          disabled={!monitorStatus.available}
          className="secondary"
        >
          Run Benchmark
        </button>
      </div>

      <div className="dashboard-panels">
        <SessionMetricsPanel metrics={metrics} />
        
        {/* Additional dashboard panels would go here */}
        <div className="dashboard-panel">
          <h2>Performance Alerts</h2>
          <div className="alerts-container">
            {alerts && alerts.length > 0 ? (
              alerts.map((alert, index) => (
                <div key={index} className={`alert alert-${alert.level}`}>
                  <h3>{alert.message}</h3>
                  <p>Source: {alert.source}</p>
                  <p>Time: {new Date(alert.timestamp).toLocaleString()}</p>
                </div>
              ))
            ) : (
              <p>No active alerts</p>
            )}
          </div>
        </div>
      </div>

      <div className="dashboard-info">
        <p>For detailed performance metrics and analytics, visit the <a href="/performance-dashboard" target="_blank">Performance Dashboard</a></p>
      </div>

      <style jsx>{`
        .dashboard-container {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
        }
        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 15px;
          border-bottom: 1px solid #eee;
        }
        .connection-status {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .status-indicator {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          display: inline-block;
        }
        .status-running {
          background-color: #28a745;
        }
        .status-stopped {
          background-color: #dc3545;
        }
        .ws-status {
          font-size: 12px;
          padding: 3px 8px;
          border-radius: 12px;
          background-color: #dc3545;
          color: white;
        }
        .ws-status.connected {
          background-color: #28a745;
        }
        .dashboard-controls {
          margin-bottom: 20px;
          display: flex;
          gap: 10px;
        }
        button {
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          background-color: #007bff;
          color: white;
          cursor: pointer;
        }
        button:hover {
          background-color: #0069d9;
        }
        button:disabled {
          background-color: #6c757d;
          cursor: not-allowed;
        }
        button.danger {
          background-color: #dc3545;
        }
        button.danger:hover {
          background-color: #c82333;
        }
        button.secondary {
          background-color: #6c757d;
        }
        button.secondary:hover {
          background-color: #5a6268;
        }
        .dashboard-panels {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: 20px;
          margin-bottom: 20px;
        }
        .dashboard-panel {
          background-color: white;
          border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          padding: 15px;
        }
        .alerts-container {
          max-height: 300px;
          overflow-y: auto;
        }
        .alert {
          padding: 10px;
          margin-bottom: 10px;
          border-radius: 4px;
        }
        .alert-info {
          background-color: #cce5ff;
          border-color: #b8daff;
        }
        .alert-warning {
          background-color: #fff3cd;
          border-color: #ffeeba;
        }
        .alert-error {
          background-color: #f8d7da;
          border-color: #f5c6cb;
        }
        .alert-critical {
          background-color: #dc3545;
          border-color: #bd2130;
          color: white;
        }
        .dashboard-info {
          margin-top: 20px;
          padding-top: 15px;
          border-top: 1px solid #eee;
          text-align: center;
        }
      `}</style>
    </div>
  );
};

export default DashboardHome;