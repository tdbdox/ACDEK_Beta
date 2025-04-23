import React from 'react';

/**
 * SessionMetricsPanel
 * UI card component for displaying session-level metrics like timing, confidence, and hallucination metrics.
 */
const SessionMetricsPanel = ({ metrics }) => {
  const formatNumber = (value, type) => {
    if (value === undefined || value === null) return '-';
    
    if (type === 'percent') {
      return `${(value * 100).toFixed(2)}%`;
    } else if (type === 'time') {
      return `${value.toFixed(2)} ms`;
    } else if (type === 'decimal') {
      return value.toFixed(4);
    } else {
      return value.toString();
    }
  };
  
  const getMetricValue = (metricName, type = 'decimal') => {
    if (!metrics || !metrics.metrics) return '-';
    
    const value = metrics.metrics[metricName];
    return formatNumber(value, type);
  };
  
  const getMetricChange = (metricName) => {
    if (!metrics || !metrics.regression_metrics || !metrics.regression_metrics[metricName]) {
      return null;
    }
    
    const regMetric = metrics.regression_metrics[metricName];
    
    if (!regMetric.is_regression && Math.abs(regMetric.delta) < 0.0001) {
      return null; // No significant change
    }
    
    const delta = regMetric.delta;
    const baseline = regMetric.baseline;
    const percentChange = Math.abs(delta) / Math.max(0.0001, Math.abs(baseline)) * 100;
    
    const isImprovement = 
      (metricName.includes('hallucination') && delta > 0) ||
      (metricName.includes('confidence') && delta > 0) ||
      (metricName.includes('success') && delta > 0) ||
      (metricName.includes('latency') && delta < 0);
    
    return {
      value: delta,
      percent: percentChange,
      isImprovement: isImprovement
    };
  };
  
  const renderMetricChange = (metricName, formatType = 'decimal') => {
    const change = getMetricChange(metricName);
    if (!change) return null;
    
    const directionArrow = change.isImprovement ? '▲' : '▼';
    const changeClass = change.isImprovement ? 'positive-change' : 'negative-change';
    const changeLabel = change.isImprovement ? 'improved' : 'worse';
    
    return (
      <div className={`metric-change ${changeClass}`}>
        {directionArrow} {formatNumber(Math.abs(change.value), formatType)} ({change.percent.toFixed(2)}% {changeLabel})
      </div>
    );
  };

  return (
    <div className="metrics-panel">
      <h2>Performance Metrics</h2>
      
      {metrics ? (
        <div className="metrics-grid">
          <div className="metric-card">
            <h3>Hallucination Reduction</h3>
            <div className="metric-value">{getMetricValue('hallucination_reduction', 'percent')}</div>
            {renderMetricChange('hallucination_reduction', 'percent')}
          </div>
          
          <div className="metric-card">
            <h3>Latency Overhead</h3>
            <div className="metric-value">{getMetricValue('latency_overhead', 'time')}</div>
            {renderMetricChange('latency_overhead', 'time')}
          </div>
          
          <div className="metric-card">
            <h3>Confidence Score</h3>
            <div className="metric-value">{getMetricValue('confidence', 'percent')}</div>
            {renderMetricChange('confidence', 'percent')}
          </div>
          
          <div className="metric-card">
            <h3>Success Rate</h3>
            <div className="metric-value">{getMetricValue('success_rate', 'percent')}</div>
            {renderMetricChange('success_rate', 'percent')}
          </div>
        </div>
      ) : (
        <div className="no-metrics">
          <p>No metrics available. Run a benchmark to see metrics.</p>
        </div>
      )}
      
      <style jsx>{`
        .metrics-panel {
          background-color: white;
          border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          padding: 15px;
        }
        h2 {
          margin-top: 0;
          margin-bottom: 15px;
          border-bottom: 1px solid #eee;
          padding-bottom: 10px;
        }
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
          gap: 15px;
        }
        .metric-card {
          background-color: #f8f9fa;
          border-radius: 4px;
          padding: 10px;
        }
        .metric-card h3 {
          margin-top: 0;
          margin-bottom: 10px;
          font-size: 14px;
          font-weight: 600;
          color: #495057;
        }
        .metric-value {
          font-size: 24px;
          font-weight: bold;
          margin-bottom: 5px;
        }
        .metric-change {
          font-size: 12px;
        }
        .positive-change {
          color: #28a745;
        }
        .negative-change {
          color: #dc3545;
        }
        .no-metrics {
          background-color: #f8f9fa;
          border-radius: 4px;
          padding: 15px;
          text-align: center;
          color: #6c757d;
        }
      `}</style>
    </div>
  );
};

export default SessionMetricsPanel;